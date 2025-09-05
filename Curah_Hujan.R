# Load library yang diperlukan
library(readxl)
library(earth)
library(caret)
library(ggplot2)
library(dplyr)
library(tidyr)
library(lubridate)
library(plotly)
library(RColorBrewer)
library(viridis)
library(reshape2)
library(gridExtra)
library(purrr)
library(broom)

# 1. Membaca dan mempersiapkan data
file_path <- " E:/Jokian/NurJannah Matematika/DATA KELAS CURAH HUJAN.xlsx"

# Cek apakah file exists
if (!file.exists(file_path)) {
  stop("File tidak ditemukan. Pastikan path file benar: ", file_path)
}

# Membaca data
data <- read_excel(file_path)
colnames(data) <- c("Curah_Hujan", "Suhu_Udara", "Kecepatan_Angin", 
                   "Kelembapan_Relatif", "Penyinaran_Matahari", "Kelas")

# Data cleaning dan preprocessing
data <- data %>%
  mutate(
    # Konversi kelas ke factor dengan nama yang valid
    Kelas_Original = Kelas,
    Kelas = case_when(
      Kelas == 0 ~ "Berawan",
      Kelas == 1 ~ "Hujan_Ringan", 
      Kelas == 2 ~ "Hujan_Sedang",
      Kelas == 3 ~ "Hujan_Lebat",
      TRUE ~ "Unknown"
    ),
    Kelas = as.factor(Kelas),
    # Menambahkan kolom bulan (dummy, asumsi data berurutan)
    Bulan = rep(1:12, length.out = nrow(data))[1:nrow(data)],
    Nama_Bulan = month.abb[Bulan],
    # Normalisasi variabel untuk analisis yang lebih baik
    Curah_Hujan_Std = as.numeric(scale(Curah_Hujan)),
    Suhu_Udara_Std = as.numeric(scale(Suhu_Udara))
  )

# Pastikan tidak ada level kosong
data$Kelas <- droplevels(data$Kelas)

# Cek distribusi kelas
cat("=== DISTRIBUSI KELAS ===\n")
print(table(data$Kelas))
cat("\n")

# 2. Pembagian data training dan testing dengan berbagai perbandingan
set.seed(123)
ratios <- c(0.75, 0.8, 0.9)
results_list <- list()

for (ratio in ratios) {
  trainIndex <- createDataPartition(data$Kelas, p = ratio, list = FALSE, times = 1)
  train_data <- data[trainIndex, ]
  test_data <- data[-trainIndex, ]
  
  # Configuration untuk cross-validation
  ctrl <- trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = multiClassSummary,
    verboseIter = FALSE
  )
  
  # Train MARS model
  mars_model <- train(
    Kelas ~ Curah_Hujan + Suhu_Udara + Kecepatan_Angin + 
            Kelembapan_Relatif + Penyinaran_Matahari,
    data = train_data,
    method = "earth",
    trControl = ctrl,
    metric = "Accuracy",
    tuneGrid = expand.grid(degree = 1:2, nprune = 2:10)
  )
  
  # Prediksi dan evaluasi model
  train_predictions <- predict(mars_model, train_data)
  test_predictions <- predict(mars_model, test_data)
  
  train_conf_matrix <- confusionMatrix(train_predictions, train_data$Kelas)
  test_conf_matrix <- confusionMatrix(test_predictions, test_data$Kelas)
  
  # Ekstrak nilai GCV, MSE, dan R2 dari model
  gcv_value <- mars_model$finalModel$gcv
  rss_value <- mars_model$finalModel$rss
  rsq_value <- mars_model$finalModel$rsq
  
  # Hitung MSE
  mse_value <- rss_value / nrow(train_data)
  
  # Simpan hasil
  results_list[[as.character(ratio)]] <- list(
    train_data = train_data,
    test_data = test_data,
    model = mars_model,
    train_conf_matrix = train_conf_matrix,
    test_conf_matrix = test_conf_matrix,
    gcv = gcv_value,
    mse = mse_value,
    rsq = rsq_value,
    accuracy_train = train_conf_matrix$overall["Accuracy"],
    accuracy_test = test_conf_matrix$overall["Accuracy"]
  )
}

# 3. Menampilkan hasil untuk berbagai perbandingan
cat("=== HASIL BERBAGAI PERBANDINGAN DATA ===\n")
for (ratio in names(results_list)) {
  cat("Perbandingan:", ratio, "(Training-Testing)\n")
  cat("GCV:", round(results_list[[ratio]]$gcv, 4), "\n")
  cat("MSE:", round(results_list[[ratio]]$mse, 4), "\n")
  cat("R-squared:", round(results_list[[ratio]]$rsq, 4), "\n")
  cat("Akurasi Training:", round(results_list[[ratio]]$accuracy_train, 4), "\n")
  cat("Akurasi Testing:", round(results_list[[ratio]]$accuracy_test, 4), "\n")
  cat("Ketetapan Klasifikasi (Training):\n")
  print(results_list[[ratio]]$train_conf_matrix$byClass[, "Balanced Accuracy"])
  cat("Ketetapan Klasifikasi (Testing):\n")
  print(results_list[[ratio]]$test_conf_matrix$byClass[, "Balanced Accuracy"])
  cat("----------------------------------------\n")
}

# 4. Trial and error untuk kombinasi parameter MARS
cat("=== TRIAL AND ERROR KOMBINASI PARAMETER MARS ===\n")
combinations <- expand.grid(
  degree = 1:3,
  nprune = 2:15,
  stringsAsFactors = FALSE
)

trial_results <- data.frame()

for (i in 1:nrow(combinations)) {
  combo <- combinations[i, ]
  
  # Train model dengan kombinasi parameter
  mars_trial <- earth(
    Kelas ~ Curah_Hujan + Suhu_Udara + Kecepatan_Angin + 
            Kelembapan_Relatif + Penyinaran_Matahari,
    data = data,
    degree = combo$degree,
    nprune = combo$nprune,
    pmethod = "backward",
    glm = list(family = binomial)
  )
  
  # Prediksi dan evaluasi
  predictions <- predict(mars_trial, type = "class")
  conf_matrix <- confusionMatrix(factor(predictions), data$Kelas)
  accuracy <- conf_matrix$overall["Accuracy"]
  
  # Simpan hasil
  trial_results <- rbind(trial_results, data.frame(
    Degree = combo$degree,
    Nprune = combo$nprune,
    GCV = mars_trial$gcv,
    MSE = mars_trial$rss / nrow(data),
    R2 = mars_trial$rsq,
    Accuracy = accuracy
  ))
}

# Tampilkan hasil trial and error
cat("Hasil Trial and Error:\n")
print(trial_results[order(trial_results$GCV), ])

# Pilih model dengan GCV terkecil
best_model_idx <- which.min(trial_results$GCV)
best_model <- trial_results[best_model_idx, ]
cat("\nModel Terbaik (GCV Terkecil):\n")
print(best_model)

# 5. Uji F dan Uji t untuk model terbaik
cat("=== UJI F DAN UJI T UNTUK MODEL TERBAIK ===\n")
best_mars_model <- earth(
  Kelas ~ Curah_Hujan + Suhu_Udara + Kecepatan_Angin + 
          Kelembapan_Relatif + Penyinaran_Matahari,
  data = data,
  degree = best_model$Degree,
  nprune = best_model$Nprune,
  pmethod = "backward",
  glm = list(family = binomial)
)

# Uji F - ANOVA
cat("Uji F (ANOVA):\n")
print(anova(best_mars_model))

# Uji t - Koefisien model
cat("Uji t (Koefisien Model):\n")
coefficient_summary <- summary(best_mars_model)
print(coefficient_summary)

# 6. Visualisasi yang lebih jelas untuk klasifikasi per bulan
monthly_classification <- data %>%
  group_by(Bulan, Nama_Bulan, Kelas) %>%
  summarise(
    Count = n(),
    .groups = 'drop'
  ) %>%
  group_by(Bulan, Nama_Bulan) %>%
  mutate(
    Percentage = Count / sum(Count) * 100
  ) %>%
  ungroup()

# Pastikan urutan bulan benar
monthly_classification$Nama_Bulan <- factor(
  monthly_classification$Nama_Bulan, 
  levels = month.abb
)

# Plot distribusi kelas per bulan
p_monthly <- ggplot(monthly_classification, 
                   aes(x = Nama_Bulan, y = Percentage, fill = Kelas)) +
  geom_bar(stat = "identity", position = "stack") +
  scale_fill_manual(values = c("Berawan" = "#2E8B57", 
                               "Hujan_Ringan" = "#1f77b4", 
                               "Hujan_Sedang" = "#ff7f0e", 
                               "Hujan_Lebat" = "#d62728")) +
  labs(title = "Distribusi Klasifikasi Curah Hujan per Bulan",
       x = "Bulan",
       y = "Persentase (%)",
       fill = "Kelas Curah Hujan") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p_monthly)

# 7. Plot curah hujan rata-rata per bulan dengan klasifikasi
monthly_rain_avg <- data %>%
  group_by(Bulan, Nama_Bulan) %>%
  summarise(
    Avg_Rainfall = mean(Curah_Hujan),
    Predominant_Class = names(which.max(table(Kelas))),
    .groups = 'drop'
  )

monthly_rain_avg$Nama_Bulan <- factor(
  monthly_rain_avg$Nama_Bulan, 
  levels = month.abb
)

p_rain_avg <- ggplot(monthly_rain_avg, 
                    aes(x = Nama_Bulan, y = Avg_Rainfall, fill = Predominant_Class)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("Berawan" = "#2E8B57", 
                               "Hujan_Ringan" = "#1f77b4", 
                               "Hujan_Sedang" = "#ff7f0e", 
                               "Hujan_Lebat" = "#d62728")) +
  labs(title = "Rata-rata Curah Hujan dan Klasifikasi Dominan per Bulan",
       x = "Bulan",
       y = "Curah Hujan Rata-rata (mm)",
       fill = "Kelas Dominan") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  geom_text(aes(label = round(Avg_Rainfall, 1)), 
            vjust = -0.5, size = 3)

print(p_rain_avg)

# 8. Plot pentingnya variabel
var_imp <- varImp(best_mars_model)
p_var_imp <- ggplot(var_imp, aes(x = reorder(rownames(var_imp), Overall), y = Overall)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Pentingnya Variabel dalam Model MARS",
       x = "Variabel",
       y = "Tingkat Kepentingan") +
  theme_minimal()

print(p_var_imp)

# 9. Plot 3D interaktif
p_3d <- plot_ly(data, 
               x = ~Suhu_Udara, 
               y = ~Kelembapan_Relatif, 
               z = ~Curah_Hujan, 
               color = ~Kelas,
               colors = c("Berawan" = "#2E8B57", 
                         "Hujan_Ringan" = "#1f77b4", 
                         "Hujan_Sedang" = "#ff7f0e", 
                         "Hujan_Lebat" = "#d62728"),
               marker = list(size = 4, opacity = 0.8),
               text = ~paste("Kelas:", Kelas, 
                            "<br>Curah Hujan:", round(Curah_Hujan, 2), "mm",
                            "<br>Suhu:", round(Suhu_Udara, 2), "°C",
                            "<br>Kelembapan:", round(Kelembapan_Relatif, 2), "%",
                            "<br>Bulan:", Nama_Bulan)) %>%
  add_markers() %>%
  layout(
    scene = list(
      xaxis = list(title = "Suhu Udara (°C)"),
      yaxis = list(title = "Kelembapan Relatif (%)"),
      zaxis = list(title = "Curah Hujan (mm)")
    ),
    title = "Klasifikasi Curah Hujan berdasarkan Parameter Meteorologi"
  )

print(p_3d)

# 10. Ringkasan hasil
cat("=== RINGKASAN HASIL ANALISIS ===\n")
cat("Model terbaik memiliki parameter:\n")
cat("- Degree:", best_model$Degree, "\n")
cat("- Nprune:", best_model$Nprune, "\n")
cat("- GCV:", round(best_model$GCV, 4), "\n")
cat("- Akurasi:", round(best_model$Accuracy, 4), "\n\n")

cat("Bulan dengan klasifikasi hujan ringan dominan:\n")
hujan_ringan_months <- monthly_rain_avg %>%
  filter(Predominant_Class == "Hujan_Ringan") %>%
  pull(Nama_Bulan)
print(hujan_ringan_months)

cat("\nBulan dengan klasifikasi hujan sedang dominan:\n")
hujan_sedang_months <- monthly_rain_avg %>%
  filter(Predominant_Class == "Hujan_Sedang") %>%
  pull(Nama_Bulan)
print(hujan_sedang_months)

cat("\nBulan dengan klasifikasi hujan lebat dominan:\n")
hujan_lebat_months <- monthly_rain_avg %>%
  filter(Predominant_Class == "Hujan_Lebat") %>%
  pull(Nama_Bulan)
print(hujan_lebat_months)

cat("\nBulan dengan klasifikasi berawan dominan:\n")
berawan_months <- monthly_rain_avg %>%
  filter(Predominant_Class == "Berawan") %>%
  pull(Nama_Bulan)
print(berawan_months)

# Simpan hasil ke file untuk dokumentasi
write.csv(trial_results, "hasil_trial_mars.csv", row.names = FALSE)
write.csv(monthly_classification, "klasifikasi_per_bulan.csv", row.names = FALSE)

cat("\n=== ANALISIS SELESAI ===\n")
cat("Hasil telah disimpan ke file CSV dan ditampilkan di konsol.\n")
