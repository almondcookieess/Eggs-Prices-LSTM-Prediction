rm(list=ls())

library(shiny)
library(tidyverse)
library(timetk)
library(tidyquant)
library(tibbletime)
library(cowplot)
library(recipes)
library(rsample)
library(yardstick)
library(keras)
library(readr)
library(lubridate)
library(glue)
library(plotly)
library(DT)


#tidy_acf
tidy_acf <- function(data, value = "Price", lags = 0:20) {
  acf_values <- acf(data[[value]], lag.max = tail(lags, 1), plot = FALSE)$acf
  ret <- tibble(acf = as.numeric(acf_values)) %>%
    rowid_to_column(var = "lag") %>%
    mutate(lag = lag - 1) %>%
    filter(lag %in% lags)
  return(ret)
}


plot_split <- function(split, raw_data, expand_y_axis = TRUE, alpha = 1, size = 1, base_size = 14) {
  # Manipulate data
  train_tbl <- training(split) %>% add_column(key = "training")
  test_tbl  <- testing(split) %>% add_column(key = "testing")
  data_manipulated <- bind_rows(train_tbl, test_tbl) %>%
    as_tbl_time(index = Date) %>%
    mutate(key = fct_relevel(key, "training", "testing"))
  
  train_time_summary <- train_tbl %>% tk_index() %>% tk_get_timeseries_summary()
  test_time_summary  <- test_tbl  %>% tk_index() %>% tk_get_timeseries_summary()
  
  # PLOT: tampilkan actual di belakang, baru training & testing
  g <- ggplot() +
    # Plot data actual
    geom_line(data = raw_data, aes(x = Date, y = Price), 
              color = "grey70", size = 0.5, alpha = 0.9) +
    # Plot training/testing overlay
    geom_line(data = data_manipulated, aes(x = Date, y = Price, color = key),
              size = size, alpha = alpha) +
    theme_tq(base_size = base_size) +
    scale_color_tq() +
    labs(
      title    = glue("Split: {split$id}"),
      subtitle = glue("{train_time_summary$start} to {test_time_summary$end}"),
      y = "", x = ""
    ) +
    theme(legend.position = "none")
  
  if (expand_y_axis) {
    raw_data_time_summary <- raw_data %>%
      tk_index() %>%
      tk_get_timeseries_summary()
    g <- g + scale_x_date(limits = c(raw_data_time_summary$start, raw_data_time_summary$end))
  }
  return(g)
}


plot_sampling_plan <- function(sampling_tbl, raw_data, expand_y_axis = TRUE, 
                               ncol = 3, alpha = 1, size = 1, base_size = 14, 
                               title = "Sampling Plan") {
  sampling_tbl_with_plots <- sampling_tbl %>%
    mutate(gg_plots = map(splits, ~plot_split(.x, raw_data, expand_y_axis, alpha, size, base_size)))
  
  plot_list <- sampling_tbl_with_plots$gg_plots 
  p_temp <- plot_list[[1]] + theme(legend.position = "bottom")
  legend <- get_legend(p_temp)
  p_body  <- plot_grid(plotlist = plot_list, ncol = ncol)
  p_title <- ggdraw() + 
    draw_label(title, size = 18, fontface = "bold", colour = palette_light()[[1]])
  g <- plot_grid(p_title, p_body, legend, ncol = 1, rel_heights = c(0.05, 1, 0.05))
  return(g)
}


# Fungsi untuk membuat sequence data
create_sequences <- function(data, tsteps) {
  # Scaling
  scaler <- scale(data$Price)
  price_scaled <- as.vector(scaler)
  attr_list <- attributes(scaler)
  
  X <- list()
  y <- c()
  for (i in 1:(length(price_scaled) - tsteps)) {
    X[[i]] <- price_scaled[i:(i + tsteps - 1)]
    y <- c(y, price_scaled[i + tsteps])
  }
  list(
    X = array(unlist(X), dim = c(length(X), tsteps, 1)),
    y = y,
    scaler = attr_list # simpan mean dan sd
  )
}

# Fungsi membangun dan melatih LSTM
build_and_train_lstm <- function(train_data, tsteps, epochs) {
  model <- keras_model_sequential() %>%
    layer_lstm(units = 64, input_shape = c(tsteps, 1), return_sequences = TRUE) %>%
    layer_dropout(rate = 0.2) %>%
    layer_lstm(units = 32, return_sequences = FALSE) %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 1) # tidak perlu aktivasi untuk regersi
  
  model %>% compile(
    loss = "mse",
    optimizer = optimizer_adam(learning_rate = 0.001),
    metrics = c("mae")
  )
  
  history <- model %>% fit(
    x = train_data$X,
    y = train_data$y,
    epochs = epochs,
    batch_size = 16,
    validation_split = 0.2,
    verbose = 0
  )
  
  list(model = model, history = history)
}

# Fungsi prediksi LSTM dengan inverse scaling
predict_keras_lstm <- function(data, tsteps, epochs) {
  sequences <- create_sequences(data, tsteps)
  model_info <- build_and_train_lstm(sequences, tsteps, epochs)
  model <- model_info$model
  
  predictions_scaled <- model %>% predict(sequences$X)
  
  # Inverse transform
  scaler_mean <- as.numeric(sequences$scaler$`scaled:center`)
  scaler_sd   <- as.numeric(sequences$scaler$`scaled:scale`)
  
  pred <- as.vector(predictions_scaled) * scaler_sd + scaler_mean
  actual <- as.vector(sequences$y) * scaler_sd + scaler_mean
  
  mae <- mean(abs(actual - pred))
  mse <- mean((actual - pred)^2)
  rmse <- sqrt(mse)
  mape <- mean(abs((actual - pred)/actual)) * 100
  
  list(
    predictions = tibble(
      Date = data$Date[(tsteps + 1):nrow(data)],
      Actual = actual,
      Predicted = pred
    ),
    metrics = tibble(
      MAE = mae,
      MSE = mse,
      RMSE = rmse,
      MAPE = mape
    ),
    history = model_info$history
  )
}

# Fungsi prediksi masa depan dengan scaling
predict_keras_lstm_future <- function(data, tsteps, epochs, forecast_period) {
  sequences <- create_sequences(data, tsteps)
  model_info <- build_and_train_lstm(sequences, tsteps, epochs)
  model <- model_info$model
  
  scaler <- sequences$scaler
  price_scaled <- as.vector(scale(data$Price))
  
  last_sequence <- tail(price_scaled, tsteps)
  future_predictions <- numeric(forecast_period)
  
  for (i in 1:forecast_period) {
    input_seq <- array(last_sequence, dim = c(1, tsteps, 1))
    prediction <- model %>% predict(input_seq)
    future_predictions[i] <- prediction
    last_sequence <- c(last_sequence[-1], prediction)
  }
  
  # Inverse transform hasil prediksi
  scaler_mean <- as.numeric(sequences$scaler$`scaled:center`)
  scaler_sd   <- as.numeric(sequences$scaler$`scaled:scale`)
  preds <- as.vector(future_predictions) * scaler_sd + scaler_mean
  
  future_dates <- seq(max(data$Date) + 7, by = "week", length.out = forecast_period)
  
  list(
    predictions = tibble(
      Date = future_dates,
      Price = preds,
      key = "predict"
    ),
    model = model,
    history = model_info$history
  )
}

# UI dengan tema pink
ui <- fluidPage(
  titlePanel(
    div(
      "Time Series Forecasting with Stateful LSTM",
      tags$br(),
      tags$span("by Syifa-Jaya-Winarni", style = "font-size:16px; color:#212121; font-weight:bold;")
    ),
    windowTitle = "Sky Blue LSTM Dashboard"
  ),
  tags$head(
    tags$style(HTML("
    body {
      background-color: #e3f2fd;
      font-family: 'Arial', sans-serif;
    }
    .navbar-default {
      background-color: #4fc3f7 !important;
      border-color: #4fc3f7;
    }
    .navbar-default .navbar-brand,
    .navbar-default .navbar-nav > li > a {
      color: #212121 !important;
      font-weight: bold;
    }
    .well {
      background-color: #81d4fa;
      border-color: #4fc3f7;
    }
    .tab-content {
      background-color: white;
      padding: 15px;
      border-radius: 5px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    .nav-tabs {
      border-bottom: 2px solid #4fc3f7;
    }
    .nav-tabs > li.active > a,
    .nav-tabs > li.active > a:hover,
    .nav-tabs > li.active > a:focus {
      background-color: #4fc3f7;
      color: #212121;
      border: 1px solid #4fc3f7;
      border-bottom-color: transparent;
    }
    .nav-tabs > li > a:hover {
      background-color: #81d4fa;
      color: #212121;
      border-color: #81d4fa;
    }
    .btn-primary {
      background-color: #4fc3f7;
      border-color: #0288d1;
      color: #212121;
      font-weight: bold;
    }
    .btn-primary:hover {
      background-color: #0288d1;
      border-color: #01579b;
      color: #fff;
    }
    .shiny-output-error {
      color: #d32f2f;
    }
    h1, h2, h3, h4 {
      color: #212121;
    }
    label, .control-label, .checkbox label, .radio label {
      color: #212121;
    }
    /* Tweak DT table if needed */
    table.dataTable {
      background-color: #e3f2fd;
      color: #212121;
    }
  "))
  ),
  
  sidebarLayout(
    sidebarPanel(
      width = 3,
      fileInput("file", "Upload CSV File", accept = c(".csv"),
                buttonLabel = "Browse...",
                placeholder = "No file selected"),
      hr(),
      h4("Data Parameters", style = "color: #d80073;"),
      numericInput("a", "Training Periods (weeks):", value = 52, min = 1),
      numericInput("b", "Testing Periods (weeks):", value = 4, min = 1),
      numericInput("c", "Skip Span (weeks):", value = 8, min = 1),
      hr(),
      h4("Model Parameters", style = "color: #d80073;"),
      numericInput("tsteps_input", "Time Steps:", value = 4, min = 1),
      numericInput("epochs", "Number of Epochs:", value = 100, min = 1),
      numericInput("forecast_period", "Forecast Weeks:", value = 4, min = 1),
      actionButton("run", "Run Analysis", class = "btn-primary"),
      hr(),
      helpText("Note: Data should have 'Date' and 'Price' columns with weekly frequency.",
               style = "color: #d80073;")
    ),
    mainPanel(
      width = 9,
      tabsetPanel(
        type = "tabs",
        tabPanel("Data Overview",
                 plotlyOutput("eda_plot"),
                 hr(),
                 plotlyOutput("seasonal_plot")),
        tabPanel("ACF Analysis", plotlyOutput("acf_plot")),
        tabPanel("Backtesting", plotOutput("backtest_plot", height = "2000px")),
        tabPanel("LSTM Training",
                 plotlyOutput("lstm_plot"),
                 plotlyOutput("mae_plot")),
        tabPanel("Forecast",
                 plotlyOutput("future_plot"),
                 h4("Forecast Table", style = "color: #d80073;"),
                 DTOutput("forecast_table")),
        tabPanel("Model Summary",
                 h4("Training Metrics", style = "color: #d80073;"),
                 tableOutput("model_metrics"),
                 h4("Model Architecture", style = "color: #d80073;"),
                 verbatimTextOutput("model_summary"),
                 h4("Training History", style = "color: #d80073;"),
                 plotlyOutput("history_plot"))
      )
    )
  )
)

# Server (sama seperti sebelumnya)
# Server
server <- function(input, output) {
  
  # Load and preprocess data
  data <- reactive({
    req(input$file)
    
    raw_data <- read_delim(input$file$datapath, delim = ";", show_col_types = FALSE)
    
    if (!all(c("Date", "Price") %in% names(raw_data))) {
      stop("CSV file must have 'Date' and 'Price' columns.")
    }
    
    raw_data %>%
      mutate(
        Date = dmy(gsub(" ", "", Date)),
        Week = week(Date),
        Month = month(Date),
        Year = year(Date)
      ) %>%
      arrange(Date) %>%
      as_tbl_time(index = Date)
  })
  
  # EDA Plot (tetap sama)
  output$eda_plot <- renderPlotly({
    req(data())
    # ... (kode plot EDA Anda) ...
    p <- data() %>%
      ggplot(aes(Date, Price)) +
      geom_line(color = "#66aed6") +
      geom_point(color = "#66aed6", size = 1) +
      labs(title = "Time Series Data", x = "Date", y = "Price") +
      theme_minimal() +
      theme(
        panel.background = element_rect(fill = "#e3f2fd"),
        plot.background = element_rect(fill = "#e3f2fd"),
        panel.grid.major = element_line(color = "#b6d4fe"),
        panel.grid.minor = element_line(color = "#b6d4fe"),
        axis.text = element_text(color = "#22577a"),
        axis.title = element_text(color = "#22577a"),
        plot.title = element_text(color = "#22577a"),
        legend.text = element_text(color = "#22577a"),
        legend.title = element_text(color = "#22577a"),
        strip.text = element_text(color = "#22577a")
      )
    ggplotly(p)
  })
  
  # Seasonal Plot (tetap sama)
  output$seasonal_plot <- renderPlotly({
    req(data())
    # ... (kode plot seasonal Anda) ...
    p <- data() %>%
      mutate(Year = factor(Year)) %>%
      ggplot(aes(Week, Price, color = Year)) +
      geom_line() +
      labs(title = "Seasonal Patterns by Year", x = "Week of Year", y = "Price") +
      theme_minimal() +
      theme(
        panel.background = element_rect(fill = "#e3f2fd"),
        plot.background = element_rect(fill = "#e3f2fd"),
        panel.grid.major = element_line(color = "#b6d4fe"),
        panel.grid.minor = element_line(color = "#b6d4fe"),
        axis.text = element_text(color = "#22577a"),
        axis.title = element_text(color = "#22577a"),
        plot.title = element_text(color = "#22577a"),
        legend.text = element_text(color = "#22577a"),
        legend.title = element_text(color = "#22577a"),
        strip.text = element_text(color = "#22577a")
      )
    ggplotly(p)
  })
  
  # ACF Plot (tetap sama)
  output$acf_plot <- renderPlotly({
    req(data())
    max_lag <- min(52, nrow(data())-1)
    acf_data <- tidy_acf(data(), value = "Price", lags = 0:max_lag)
    p <- ggplot(acf_data, aes(lag, acf)) +
      geom_segment(aes(xend = lag, yend = 0), color = "#66aed6") +
      geom_hline(yintercept = 0) +
      geom_hline(yintercept = c(-1, 1)*1.96/sqrt(nrow(data())),
                 linetype = "dashed", color = "red") +
      labs(title = "Autocorrelation Function (ACF)", x = "Lag (weeks)", y = "ACF") +
      theme_minimal() +
      theme(
        panel.background = element_rect(fill = "#e3f2fd"),
        plot.background = element_rect(fill = "#e3f2fd"),
        panel.grid.major = element_line(color = "#b6d4fe"),
        panel.grid.minor = element_line(color = "#b6d4fe"),
        axis.text = element_text(color = "#22577a"),
        axis.title = element_text(color = "#22577a"),
        plot.title = element_text(color = "#22577a"),
        legend.text = element_text(color = "#22577a"),
        legend.title = element_text(color = "#22577a"),
        strip.text = element_text(color = "#22577a")
      )
    ggplotly(p)
  })
  
  
  # Backtesting Plot (tetap sama)
  output$backtest_plot <- renderPlot({
    req(data(), input$a, input$b, input$c)
    periods_train <- input$a
    periods_test  <- input$b
    skip_span     <- input$c
    
    rolling_origin_resamples <- rolling_origin(
      data(),
      initial    = periods_train,
      assess     = periods_test,
      cumulative = FALSE,
      skip       = skip_span
    )
    
    plot_sampling_plan(
      rolling_origin_resamples,
      raw_data = data(),         # <---- Ini WAJIB
      expand_y_axis = TRUE,
      ncol = 4,
      alpha = 1,
      size = 1,
      base_size = 10,
      title = "Backtesting Strategy: Rolling Origin Sampling Plan"
    )
  })
  
  
  # LSTM Results
  lstm_results <- reactive({
    req(input$run, data(), input$tsteps_input, input$epochs)
    
    predict_keras_lstm(
      data = data(),
      tsteps = input$tsteps_input,
      epochs = input$epochs
    )
  })
  
  # LSTM Plot (tetap sama)
  output$lstm_plot <- renderPlotly({
    results <- lstm_results()
    
    p <- results$predictions %>%
      pivot_longer(cols = c(Actual, Predicted), names_to = "Type", values_to = "Price") %>%
      ggplot(aes(Date, Price, color = Type)) +
      geom_line() +
      scale_color_manual(values = c("Actual" = "#66aed6", "Predicted" = "#ffc107")) +
      labs(title = "Actual vs Predicted Values", x = "Date", y = "Price") +
      theme_minimal() +
      theme(panel.background = element_rect(fill = "#e3f2fd"),
            plot.background = element_rect(fill = "#e3f2fd"),
            panel.grid.major = element_line(color = "#b6d4fe"),
            panel.grid.minor = element_line(color = "#b6d4fe"),
            axis.text = element_text(color = "#22577a"),
            axis.title = element_text(color = "#22577a"),
            plot.title = element_text(color = "#22577a"),
            strip.text = element_text(color = "#22577a"))
    ggplotly(p)
  })
  
  # MAE vs Epoch Plot
  output$mae_plot <- renderPlotly({
    results <- lstm_results()
    
    # Periksa apakah objek history ada dan merupakan list
    if (!is.null(results$history) && is.list(results$history)) {
      history_df <- as.data.frame(results$history)
      
      # Periksa apakah kolom 'metrics_mae' dan 'val_metrics_mae' ada
      if ("metrics_mae" %in% names(history_df) && "val_metrics_mae" %in% names(history_df)) {
        p <- ggplot(history_df, aes(x = epoch)) +
          geom_line(aes(y = metrics_mae, color = "Training MAE")) +
          geom_line(aes(y = val_metrics_mae, color = "Validation MAE")) +
          labs(title = "MAE vs Epoch", x = "Epoch", y = "MAE") +
          scale_color_manual(values = c("Training MAE" = "#66aed6", "Validation MAE" = "#ffc107")) +
          theme_minimal() +
          theme(panel.background = element_rect(fill = "#e3f2fd"),
                plot.background = element_rect(fill = "#e3f2fd"),
                panel.grid.major = element_line(color = "#b6d4fe"),
                panel.grid.minor = element_line(color = "#b6d4fe"),
                axis.text = element_text(color = "#22577a"),
                axis.title = element_text(color = "#22577a"),
                plot.title = element_text(color = "#22577a"),
                strip.text = element_text(color = "#22577a"))
        ggplotly(p)
      } else {
        # Kembalikan plot kosong atau pesan error jika kolom tidak ditemukan
        plotly::plotly_empty(type = "scatter", mode = "markers") %>%
          layout(title = "MAE data not available")
      }
    } else {
      # Kembalikan plot kosong atau pesan error jika history tidak tersedia
      plotly::plotly_empty(type = "scatter", mode = "markers") %>%
        layout(title = "Training history not available")
    }
  })
  
  # Training History Plot
  output$history_plot <- renderPlotly({
    results <- lstm_results()
    
    # Pastikan objek history adalah list Keras
    history <- results$history
    
    # Cek dan extract
    if (!is.null(history) && !is.null(history$params)) {
      epochs <- seq_len(history$params$epochs)
      loss <- as.numeric(history$metrics$loss)
      val_loss <- as.numeric(history$metrics$val_loss)
      df <- data.frame(epoch = epochs, loss = loss)
      if (!is.null(val_loss)) df$val_loss <- val_loss
      
      df_long <- df %>%
        pivot_longer(cols = c("loss", "val_loss"), names_to = "Type", values_to = "Value")
      
      p <- ggplot(df_long, aes(x = epoch, y = Value, color = Type)) +
        geom_line(size = 1.2) +
        scale_color_manual(values = c("loss" = "#4fc3f7", "val_loss" = "#212121")) +
        labs(title = "Training History", x = "Epoch", y = "Loss") +
        theme_minimal() +
        theme(panel.background = element_rect(fill = "#e3f2fd"),
              plot.background = element_rect(fill = "#e3f2fd"),
              axis.text = element_text(color = "#212121"),
              axis.title = element_text(color = "#212121"),
              plot.title = element_text(color = "#212121"),
              legend.text = element_text(color = "#212121"),
              legend.title = element_text(color = "#212121"))
      ggplotly(p)
    } else {
      plotly::plotly_empty(type = "scatter", mode = "markers") %>%
        layout(title = "Loss data not available")
    }
  })
  
  
  # Future Predictions (tetap sama)
  future_results <- reactive({
    req(input$run, data(), input$tsteps_input, input$epochs, input$forecast_period)
    # ... (kode prediksi masa depan Anda) ...
    predict_keras_lstm_future(
      data = data(),
      tsteps = input$tsteps_input,
      epochs = input$epochs,
      forecast_period = input$forecast_period
    )
  })
  
  # Future Plot (tetap sama)
  output$future_plot <- renderPlotly({
    results <- future_results()
    # ... (kode plot masa depan Anda) ...
    plot_data <- data() %>%
      mutate(key = "actual") %>%
      bind_rows(results$predictions)
    p <- plot_data %>%
      ggplot(aes(Date, Price, color = key)) +
      geom_line(data = filter(plot_data, key == "actual"), color = "#66aed6") +
      geom_point(data = filter(plot_data, key == "predict"), color = "#ffc107", size = 0.5) +
      geom_line(data = filter(plot_data, key == "predict"), color = "#ffc107") +
      scale_color_manual(values = c("actual" = "#66aed6", "predict" = "#ffc107")) +
      labs(title = glue("Price Forecast: {input$forecast_period} Weeks Ahead"),
           color = "") +
      theme_minimal() +
      theme(panel.background = element_rect(fill = "#e3f2fd"),
            plot.background = element_rect(fill = "#e3f2fd"),
            panel.grid.major = element_line(color = "#b6d4fe"),
            panel.grid.minor = element_line(color = "#b6d4fe"),
            axis.text = element_text(color = "#22577a"),
            axis.title = element_text(color = "#22577a"),
            plot.title = element_text(color = "#22577a"),
            strip.text = element_text(color = "#22577a"))
    ggplotly(p)
  })
  
  # Forecast Table (tetap sama)
  output$forecast_table <- renderDT({
    results <- future_results()
    # ... (kode tabel forecast Anda) ...
    datatable(
      results$predictions %>%
        mutate(
          Date = format(Date, "%Y-%m-%d"),
          Price = round(Price, 2)
        ) %>%
        rename(
          "Forecast Date" = Date,
          "Forecast Price" = Price
        ),
      options = list(
        pageLength = 16,
        lengthMenu = c(5, 10, 16, 20),
        dom = 't'
      ),
      rownames = FALSE
    ) %>%
      formatStyle(columns = c(1, 2), fontSize = '12px')
  })
  
  # Model Metrics (tetap sama)
  output$model_metrics <- renderTable({
    results <- lstm_results()
    # ... (kode metrik model Anda) ...
    metrics <- results$metrics %>%
      mutate(across(everything(), ~round(., 3))) %>%
      pivot_longer(everything(), names_to = "Metric", values_to = "Value")
    metrics
  }, striped = TRUE, bordered = TRUE)
  
  # Model Summary (tetap sama)
  output$model_summary <- renderPrint({
    results <- future_results()
    # ... (kode summary model Anda) ...
    if (!is.null(results$model)) {
      summary(results$model)
    } else {
      "Model summary not available."
    }
  })
}
# Run the Shiny App
shinyApp(ui = ui, server = server)
