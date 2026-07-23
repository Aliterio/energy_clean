# Appliance energy prediction - full analysis pipeline
# Data: Candanedo et al. (2017), UCI / Kaggle "Appliances Energy Prediction"
# Author: Ali Bahmanyar
#
# Pipeline: clean data -> EDA -> fit 6 models (time-ordered split) ->
# SHAP on the winning model -> residual diagnostics.
#
# Packages needed:
# install.packages(c("dplyr", "lubridate", "zoo", "ggplot2", "corrplot",
#                     "gridExtra", "reshape2", "scales", "viridis",
#                     "glmnet", "randomForest", "gbm", "e1071", "tidyr"))
# remotes::install_github("bgreenwell/fastshap")   # not on CRAN yet

library(dplyr)
library(lubridate)
library(ggplot2)
library(gridExtra)
library(scales)

set.seed(2026)

raw_path <- "data/energydata_complete.csv"
clean_path <- "data/energy_clean.csv"
fig_dir <- "figures"
out_dir <- "output"
mod_dir <- "models"

for (d in c(fig_dir, out_dir, mod_dir)) dir.create(d, showWarnings = FALSE)

theme_pub <- theme_bw(base_size = 12) +
  theme(panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold", size = 13),
        axis.title = element_text(size = 12),
        legend.position = "right")

## =========================================================================
## 1. Data cleaning + feature engineering
## =========================================================================

df <- read.csv(raw_path, stringsAsFactors = FALSE)
df$date <- ymd_hms(df$date, tz = "UTC")

stopifnot(!any(is.na(df$date)))
stopifnot(sum(is.na(df)) == 0)

# a handful of duplicate timestamps show up in the raw file
df <- df %>% arrange(date) %>% distinct(date, .keep_all = TRUE)

# rv1/rv2 are random control columns the original authors added for
# feature-selection testing - not real predictors, drop them
df$rv1 <- NULL
df$rv2 <- NULL

df <- df %>%
  mutate(
    hour = hour(date),
    dow = wday(date, label = FALSE, week_start = 1),
    is_weekend = as.integer(dow %in% c(6, 7)),
    month = month(date),
    tod_sin = sin(2 * pi * hour / 24),
    tod_cos = cos(2 * pi * hour / 24),
    dow_sin = sin(2 * pi * dow / 7),
    dow_cos = cos(2 * pi * dow / 7)
  )

rh_cols <- grep("^RH_", names(df), value = TRUE)
t_cols <- grep("^T[0-9]", names(df), value = TRUE)

df$RH_avg <- rowMeans(df[, rh_cols])
df$T_avg <- rowMeans(df[, t_cols])
df$T_range <- apply(df[, t_cols], 1, max) - apply(df[, t_cols], 1, min)
df$T_out_diff <- df$T_avg - df$T_out

# appliance use is a time series, so lag/rolling features matter a lot here
df <- df %>%
  mutate(
    Appliances_lag1 = lag(Appliances, 1),
    Appliances_lag6 = lag(Appliances, 6),
    Appliances_roll6 = zoo::rollapply(Appliances, width = 6, FUN = mean,
                                       align = "right", fill = NA)
  )

df <- df %>% filter(!is.na(Appliances_lag6) & !is.na(Appliances_roll6))

# cap extreme spikes at the 99.5th pct, then log1p to tame the skew
cap <- quantile(df$Appliances, 0.995)
n_capped <- sum(df$Appliances > cap)
df$Appliances_capped <- pmin(df$Appliances, cap)
df$log_Appliances <- log1p(df$Appliances_capped)

n_total <- nrow(df)
n_train <- floor(0.70 * n_total)
n_val <- floor(0.15 * n_total)

# time-ordered split, NOT random - this matters a lot for a time series
df$split <- c(rep("train", n_train), rep("val", n_val),
               rep("test", n_total - n_train - n_val))

# write.csv silently drops the time part of a POSIXct when it's exactly
# midnight, which corrupts a chunk of timestamps on reload - format it
# explicitly to avoid that
df$date <- format(df$date, "%Y-%m-%d %H:%M:%S")

write.csv(df, clean_path, row.names = FALSE)

cleaning_summary <- data.frame(
  metric = c("n_rows_raw", "n_rows_clean", "n_duplicates_removed",
             "n_capped_outliers", "cap_995_value", "n_train", "n_val", "n_test"),
  value = c(19735, n_total, 19735 - n_total, n_capped, round(cap, 2),
            n_train, n_val, n_total - n_train - n_val)
)
write.csv(cleaning_summary, file.path(out_dir, "cleaning_summary.csv"), row.names = FALSE)

## =========================================================================
## 2. Exploratory plots
## =========================================================================

df_plot <- read.csv(clean_path, stringsAsFactors = FALSE)
df_plot$date <- ymd_hms(df_plot$date, tz = "UTC")

p1a <- ggplot(df_plot, aes(x = date, y = Appliances)) +
  geom_line(color = "#2c3e50", linewidth = 0.25) +
  labs(title = "(a) Full study period", x = NULL, y = "Appliances (Wh)") +
  theme_pub

wk <- df_plot %>% filter(date >= min(date) & date < min(date) + days(7))
p1b <- ggplot(wk, aes(x = date, y = Appliances)) +
  geom_line(color = "#c0392b", linewidth = 0.4) +
  labs(title = "(b) One-week detail", x = "Date", y = "Appliances (Wh)") +
  theme_pub

fig1 <- grid.arrange(p1a, p1b, nrow = 2)
ggsave(file.path(fig_dir, "fig1_timeseries.png"), fig1, width = 7.5, height = 6, dpi = 300)

p2a <- ggplot(df_plot, aes(x = Appliances)) +
  geom_histogram(bins = 50, fill = "#2980b9", color = "white") +
  labs(title = "(a) Raw Appliances", x = "Appliances (Wh)", y = "Count") +
  theme_pub

p2b <- ggplot(df_plot, aes(x = log_Appliances)) +
  geom_histogram(bins = 50, fill = "#27ae60", color = "white") +
  labs(title = "(b) log(1 + Appliances)", x = "log(1 + Appliances)", y = "Count") +
  theme_pub

fig2 <- grid.arrange(p2a, p2b, ncol = 2)
ggsave(file.path(fig_dir, "fig2_distribution.png"), fig2, width = 8, height = 4, dpi = 300)

num_cols <- c("Appliances", "lights", "T1", "RH_1", "T2", "RH_2", "T3", "RH_3",
              "T4", "RH_4", "T5", "RH_5", "T6", "RH_6", "T7", "RH_7", "T8",
              "RH_8", "T9", "RH_9", "T_out", "Press_mm_hg", "RH_out",
              "Windspeed", "Visibility", "Tdewpoint", "T_avg", "RH_avg",
              "T_range", "T_out_diff")
cmat <- cor(df_plot[, num_cols], use = "pairwise.complete.obs")

library(corrplot)
png(file.path(fig_dir, "fig3_corrplot.png"), width = 9, height = 9, units = "in", res = 300)
corrplot(cmat, method = "color", type = "upper", order = "hclust",
         tl.col = "black", tl.cex = 0.7, tl.srt = 45,
         col = colorRampPalette(c("#c0392b", "white", "#2980b9"))(200),
         diag = FALSE, title = "Correlation matrix of numeric predictors",
         mar = c(0, 0, 2, 0))
dev.off()

hr <- df_plot %>%
  mutate(day_type = ifelse(is_weekend == 1, "Weekend", "Weekday")) %>%
  group_by(hour, day_type) %>%
  summarise(mean_app = mean(Appliances), se = sd(Appliances) / sqrt(n()), .groups = "drop")

fig4 <- ggplot(hr, aes(x = hour, y = mean_app, color = day_type, fill = day_type)) +
  geom_ribbon(aes(ymin = mean_app - se, ymax = mean_app + se), alpha = 0.2, color = NA) +
  geom_line(linewidth = 0.9) +
  scale_color_manual(values = c(Weekday = "#2c3e50", Weekend = "#e67e22")) +
  scale_fill_manual(values = c(Weekday = "#2c3e50", Weekend = "#e67e22")) +
  scale_x_continuous(breaks = seq(0, 23, 4)) +
  labs(title = "Mean appliance energy use by hour of day", x = "Hour of day",
       y = "Mean Appliances (Wh)", color = NULL, fill = NULL) +
  theme_pub
ggsave(file.path(fig_dir, "fig4_hourly_profile.png"), fig4, width = 7, height = 4.5, dpi = 300)

library(viridis)
fig5 <- ggplot(df_plot, aes(x = T_out_diff, y = Appliances)) +
  geom_hex(bins = 40) +
  scale_fill_viridis(option = "magma", trans = "log10", name = "Count") +
  labs(title = "Appliances vs indoor-outdoor temperature differential",
       x = expression(T[avg] - T[out] ~ (degree * C)), y = "Appliances (Wh)") +
  theme_pub
ggsave(file.path(fig_dir, "fig5_hexbin_tdiff.png"), fig5, width = 7, height = 5, dpi = 300)

## =========================================================================
## 3. Models: 3 classical stats + 3 ML, tuned on the validation split
## =========================================================================

library(glmnet)
library(randomForest)
library(gbm)
library(e1071)

drop_cols <- c("date", "Appliances", "Appliances_capped", "log_Appliances", "split")
feat_cols <- setdiff(names(df_plot), drop_cols)

tr <- df_plot %>% filter(split == "train")
va <- df_plot %>% filter(split == "val")
te <- df_plot %>% filter(split == "test")
trva <- bind_rows(tr, va)

y_tr <- tr$log_Appliances
y_va <- va$log_Appliances
y_trva <- trva$log_Appliances
y_te <- te$log_Appliances

x_tr <- as.matrix(tr[, feat_cols])
x_va <- as.matrix(va[, feat_cols])
x_trva <- as.matrix(trva[, feat_cols])
x_te <- as.matrix(te[, feat_cols])

fmla <- as.formula(paste("log_Appliances ~", paste(feat_cols, collapse = " + ")))

rmse <- function(actual, pred) sqrt(mean((actual - pred)^2))
mae <- function(actual, pred) mean(abs(actual - pred))
r_sq <- function(actual, pred) 1 - sum((actual - pred)^2) / sum((actual - mean(actual))^2)

eval_wh <- function(pred_log, actual_log, cap) {
  pred_wh <- pmin(expm1(pred_log), cap)
  actual_wh <- expm1(actual_log)
  c(RMSE = rmse(actual_wh, pred_wh), MAE = mae(actual_wh, pred_wh), R2 = r_sq(actual_wh, pred_wh))
}

cap_val <- max(df_plot$Appliances_capped)

# --- OLS ---
fit_ols <- lm(fmla, data = trva)
pred_ols_te <- predict(fit_ols, newdata = te)

# --- Ridge ---
lam_grid <- 10^seq(-4, 1, length.out = 30)
ridge_tune <- glmnet(x_tr, y_tr, alpha = 0, lambda = lam_grid, standardize = TRUE)
ridge_va_err <- apply(predict(ridge_tune, newx = x_va), 2, function(p) rmse(y_va, p))
lambda_ridge <- lam_grid[which.min(ridge_va_err)]
fit_ridge <- glmnet(x_trva, y_trva, alpha = 0, lambda = lambda_ridge, standardize = TRUE)
pred_ridge_te <- as.numeric(predict(fit_ridge, newx = x_te))

# --- Lasso ---
lasso_tune <- glmnet(x_tr, y_tr, alpha = 1, lambda = lam_grid, standardize = TRUE)
lasso_va_err <- apply(predict(lasso_tune, newx = x_va), 2, function(p) rmse(y_va, p))
lambda_lasso <- lam_grid[which.min(lasso_va_err)]
fit_lasso <- glmnet(x_trva, y_trva, alpha = 1, lambda = lambda_lasso, standardize = TRUE)
pred_lasso_te <- as.numeric(predict(fit_lasso, newx = x_te))

# --- Random forest ---
mtry_grid <- unique(pmax(c(5, floor(sqrt(length(feat_cols))), floor(length(feat_cols) / 3)), 2))
rf_va_err <- sapply(mtry_grid, function(m) {
  f <- randomForest(x = x_tr, y = y_tr, ntree = 300, mtry = m)
  rmse(y_va, predict(f, x_va))
})
mtry_best <- mtry_grid[which.min(rf_va_err)]
fit_rf <- randomForest(x = x_trva, y = y_trva, ntree = 500, mtry = mtry_best, importance = TRUE)
pred_rf_te <- predict(fit_rf, x_te)

# --- GBM ---
gbm_tune <- gbm(fmla, data = tr, distribution = "gaussian", n.trees = 2000,
                 interaction.depth = 4, shrinkage = 0.01, cv.folds = 0, verbose = FALSE)
gbm_va_pred <- predict(gbm_tune, newdata = va, n.trees = 1:2000)
gbm_va_err <- apply(gbm_va_pred, 2, function(p) rmse(y_va, p))
ntree_best <- which.min(gbm_va_err)
fit_gbm <- gbm(fmla, data = trva, distribution = "gaussian", n.trees = ntree_best,
               interaction.depth = 4, shrinkage = 0.01, cv.folds = 0, verbose = FALSE)
pred_gbm_te <- predict(fit_gbm, newdata = te, n.trees = ntree_best)

# --- SVM (radial) ---
# kernel SVMs get painfully slow on ~14k rows, so tuning + the final fit
# run on a random subsample - noted as a limitation in the writeup
svm_n <- min(4000, nrow(tr))
tr_svm <- tr[sample(nrow(tr), svm_n), ]
svm_grid <- expand.grid(cost = c(1, 10), gamma = c(0.01, 0.1))
svm_va_err <- mapply(function(cst, gma) {
  f <- svm(fmla, data = tr_svm, kernel = "radial", cost = cst, gamma = gma)
  rmse(y_va, predict(f, va))
}, svm_grid$cost, svm_grid$gamma)
svm_best <- svm_grid[which.min(svm_va_err), ]

trva_svm <- trva[sample(nrow(trva), min(6000, nrow(trva))), ]
fit_svm <- svm(fmla, data = trva_svm, kernel = "radial", cost = svm_best$cost, gamma = svm_best$gamma)
pred_svm_te <- predict(fit_svm, te)

perf <- rbind(
  OLS = eval_wh(pred_ols_te, y_te, cap_val),
  Ridge = eval_wh(pred_ridge_te, y_te, cap_val),
  Lasso = eval_wh(pred_lasso_te, y_te, cap_val),
  RF = eval_wh(pred_rf_te, y_te, cap_val),
  GBM = eval_wh(pred_gbm_te, y_te, cap_val),
  SVM = eval_wh(pred_svm_te, y_te, cap_val)
)
perf <- data.frame(model = rownames(perf), round(perf, 3), row.names = NULL)
perf <- perf[order(perf$RMSE), ]
write.csv(perf, file.path(out_dir, "model_performance_test.csv"), row.names = FALSE)

tuned <- data.frame(
  param = c("ridge_lambda", "lasso_lambda", "rf_mtry", "gbm_ntree", "svm_cost", "svm_gamma"),
  value = c(lambda_ridge, lambda_lasso, mtry_best, ntree_best, svm_best$cost, svm_best$gamma)
)
write.csv(tuned, file.path(out_dir, "tuned_hyperparameters.csv"), row.names = FALSE)

saveRDS(fit_ols, file.path(mod_dir, "fit_ols.rds"))
saveRDS(fit_ridge, file.path(mod_dir, "fit_ridge.rds"))
saveRDS(fit_lasso, file.path(mod_dir, "fit_lasso.rds"))
saveRDS(fit_rf, file.path(mod_dir, "fit_rf.rds"))
saveRDS(fit_gbm, file.path(mod_dir, "fit_gbm.rds"))
saveRDS(fit_svm, file.path(mod_dir, "fit_svm.rds"))
saveRDS(list(feat_cols = feat_cols, cap_val = cap_val, best_ntree_gbm = ntree_best),
        file.path(mod_dir, "meta.rds"))

print(perf)

## =========================================================================
## 4. SHAP + residual diagnostics on the winning model (GBM)
## =========================================================================

library(tidyr)

pred_wrapper <- function(object, newdata) predict(object, newdata = newdata, n.trees = ntree_best)

bg <- tr[sample(nrow(tr), min(300, nrow(tr))), feat_cols]
ex_data <- te[sample(nrow(te), min(500, nrow(te))), feat_cols]

shap_vals <- fastshap::explain(fit_gbm, X = bg, pred_wrapper = pred_wrapper,
                                nsim = 50, newdata = ex_data)

mean_abs_shap <- sort(colMeans(abs(shap_vals)), decreasing = TRUE)
top_feats <- names(mean_abs_shap)[1:15]

shap_long <- as.data.frame(shap_vals[, top_feats]) %>%
  mutate(row_id = row_number()) %>%
  pivot_longer(-row_id, names_to = "feature", values_to = "shap_value")

feat_val_long <- as.data.frame(ex_data[, top_feats]) %>%
  mutate(row_id = row_number()) %>%
  pivot_longer(-row_id, names_to = "feature", values_to = "raw_value") %>%
  group_by(feature) %>%
  mutate(scaled_value = scales::rescale(raw_value, to = c(0, 1))) %>%
  ungroup()

shap_plot_df <- shap_long %>%
  left_join(feat_val_long, by = c("row_id", "feature")) %>%
  mutate(feature = factor(feature, levels = rev(top_feats)))

fig6 <- ggplot(shap_plot_df, aes(x = shap_value, y = feature, color = scaled_value)) +
  geom_jitter(height = 0.2, width = 0, alpha = 0.6, size = 1.2) +
  scale_color_gradient(low = "#2980b9", high = "#c0392b", name = "Feature\nvalue",
                        breaks = c(0, 1), labels = c("Low", "High")) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
  labs(title = "SHAP summary plot (GBM, top 15 features)",
       x = "SHAP value (impact on log-Appliances prediction)", y = NULL) +
  theme_pub
ggsave(file.path(fig_dir, "fig6_shap_summary.png"), fig6, width = 8, height = 6.5, dpi = 300)

imp_df <- data.frame(feature = names(mean_abs_shap)[1:15], mean_abs_shap = mean_abs_shap[1:15]) %>%
  mutate(feature = factor(feature, levels = rev(feature)))

fig7 <- ggplot(imp_df, aes(x = mean_abs_shap, y = feature)) +
  geom_col(fill = "#2c3e50") +
  labs(title = "Mean |SHAP value| by feature (GBM, top 15)", x = "Mean |SHAP value|", y = NULL) +
  theme_pub
ggsave(file.path(fig_dir, "fig7_shap_importance.png"), fig7, width = 7, height = 6, dpi = 300)

write.csv(imp_df, file.path(out_dir, "shap_importance.csv"), row.names = FALSE)

pred_log_te <- predict(fit_gbm, newdata = te, n.trees = ntree_best)
resid_log <- te$log_Appliances - pred_log_te

diag_df <- data.frame(fitted = pred_log_te, resid = resid_log, idx = seq_len(nrow(te)))

p_a <- ggplot(diag_df, aes(x = fitted, y = resid)) +
  geom_point(alpha = 0.3, size = 0.8, color = "#2c3e50") +
  geom_hline(yintercept = 0, color = "#c0392b", linetype = "dashed") +
  geom_smooth(se = FALSE, color = "#e67e22", linewidth = 0.8, method = "loess") +
  labs(title = "(a) Residuals vs fitted", x = "Fitted (log scale)", y = "Residual") +
  theme_pub

p_b <- ggplot(diag_df, aes(sample = resid)) +
  stat_qq(size = 0.8, alpha = 0.4, color = "#2c3e50") +
  stat_qq_line(color = "#c0392b") +
  labs(title = "(b) Normal Q-Q plot", x = "Theoretical quantiles", y = "Sample quantiles") +
  theme_pub

acf_obj <- acf(diag_df$resid, plot = FALSE, lag.max = 48)
acf_df <- data.frame(lag = acf_obj$lag[, 1, 1], acf = acf_obj$acf[, 1, 1])
ci_line <- qnorm(0.975) / sqrt(nrow(te))

p_c <- ggplot(acf_df, aes(x = lag, y = acf)) +
  geom_hline(yintercept = c(-ci_line, ci_line), linetype = "dashed", color = "#2980b9") +
  geom_segment(aes(xend = lag, yend = 0), color = "#2c3e50") +
  labs(title = "(c) ACF of residuals (time-ordered test set)", x = "Lag (10-min steps)",
       y = "Autocorrelation") +
  theme_pub

p_d <- ggplot(diag_df, aes(x = resid)) +
  geom_histogram(bins = 50, fill = "#27ae60", color = "white") +
  labs(title = "(d) Residual distribution", x = "Residual", y = "Count") +
  theme_pub

fig8 <- grid.arrange(p_a, p_b, p_c, p_d, nrow = 2, ncol = 2)
ggsave(file.path(fig_dir, "fig8_residual_diagnostics.png"), fig8, width = 9, height = 7.5, dpi = 300)

dw_stat <- sum(diff(diag_df$resid)^2) / sum(diag_df$resid^2)
lb_test <- Box.test(diag_df$resid, lag = 10, type = "Ljung-Box")
sw_test <- shapiro.test(sample(diag_df$resid, min(5000, nrow(diag_df))))

diag_summary <- data.frame(
  statistic = c("Durbin-Watson", "Ljung-Box (lag=10) p-value", "Shapiro-Wilk p-value",
                "Residual mean", "Residual SD"),
  value = c(round(dw_stat, 4), signif(lb_test$p.value, 4), signif(sw_test$p.value, 4),
            round(mean(diag_df$resid), 5), round(sd(diag_df$resid), 4))
)
write.csv(diag_summary, file.path(out_dir, "residual_diagnostics_summary.csv"), row.names = FALSE)

print(diag_summary)
