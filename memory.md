# 🧠 Memory & Knowledge Base — Toxic Comment Classification

> Tổng hợp kiến thức về cấu trúc project, các module, pipeline, config, tracking, và luồng xử lý dữ liệu.

---

## 📁 Tổng quan cấu trúc project

```text
ToxicCommentClassification/
├── config/                  # Cấu hình cho mọi module
│   ├── base_config.py       # BaseConfig (dataclass + override)
│   ├── cleaning_config.py   # Cấu hình cleaning steps
│   ├── dataset_config.py    # Tên dataset, cột, encoding
│   ├── mlflow_tracking_config.py
│   ├── path_config.py       # Đường dẫn chuẩn (local + Drive)
│   ├── preprocessing_config.py
│   ├── split_config.py      # Tỷ lệ train/val/test
│   └── validation_config.py
├── src/
│   ├── dataset/             # Core data processing
│   │   ├── cleaning.py      # Làm sạch text
│   │   ├── eda.py           # EDA & visualization
│   │   ├── feature_engineering.py  # Feature creation
│   │   ├── preprocessing.py # Text preprocessing
│   │   ├── split.py         # Train/val/test split
│   │   └── validation.py    # Data quality checks
│   ├── pipeline/            # Pipeline orchestration
│   │   ├── base_pipeline.py # BasePipeline + PipelineStep
│   │   ├── cleaning_pipeline.py
│   │   ├── preprocessing_pipeline.py
│   │   └── validation_pipeline.py
│   ├── tracking/            # MLflow tracking
│   │   ├── mlflow_tracker.py
│   │   ├── dataset_tracker.py
│   │   └── experiment_manager.py
│   └── utils/               # Utilities
│       ├── csv.py           # CSV reading
│       └── downloader.py    # Google Drive download
├── scripts/                 # Scripts (dataset, mlflow, utils)
├── datasets/                # Data storage
├── docs/                    # Documentation
├── notebooks/               # Jupyter notebooks
└── tests/                   # Unit tests
```

---

## 🏗️ Config System (`config/`)

### `base_config.py`

- **`BaseConfig[T]`**: Abstract dataclass với generic self-type.
- **`override(**kwargs) -> T`\*\*: Tạo instance mới với các field được ghi đè. Dùng để tùy chỉnh config mà không sửa object gốc.

### `dataset_config.py`

- **`DatasetConfig`**: Cấu hình dataset cụ thể.
  - `dataset_name`, `version`: Tên và version dataset.
  - `comment_col`, `label_col`: Tên cột trong CSV (mặc định: `"comment"`, `"is_toxic"`).
  - `encoding`: Encoding file CSV.

### `cleaning_config.py`

- **`CleaningConfig`**: Bật/tắt từng bước cleaning + tham số.
  - 9 flags `enable_*` cho từng bước (html, url, mention, emoji, special_chars, null/empty, non-text, duplicate, outlier).
  - `keep_punctuation`: Dấu câu giữ lại (mặc định: `.,!?`).
  - `max_null_label_ratio`: Ngưỡng null label cho phép (5%).
  - `outlier_*`: Tham số Isolation Forest (contamination, random_state, feature_cols).

### `preprocessing_config.py`

- **`PreprocessingConfig`**: Cấu hình preprocessing.
  - `enable_normalize`, `enable_tokenize`, `enable_stopword_filter`.
  - `normalize_lower`, `normalize_strip_spaces`.
  - `tokenizer`, `stopwords`: Có thể override (None = dùng default).
  - `return_tokens`: True = list token, False = string.

### `split_config.py`

- **`SplitConfig`**: Tỷ lệ split (70/15/15), stratify, random_state, shuffle.
  - `train_filename`, `val_filename`, `test_filename`: Tên file đầu ra.

### `validation_config.py`

- **`ValidationConfig`**: Bật/tắt các bước kiểm tra.
  - `required_cols`: Danh sách cột bắt buộc (mặc định: `["comment", "is_toxic"]`).
  - 4 flags `enable_*`: column_check, null_check, empty_or_no_letter_check, duplicate_check.

### `path_config.py`

- **`PathConfig`**: Đường dẫn chuẩn cho project.
  - `project_root`: Tự động lấy từ vị trí file config.
  - `data_dir`: `"datasets"`.
  - `drive_mount_path`, `drive_data_dir`: Dành cho Google Colab.
  - Các helper method: `get_dataset_dir()`, `get_version_dir()`, `get_raw_dir()`, `get_processed_dir()`, `get_split_dir()`, `get_meta_dir()`.
  - `get_processed_filename(raw_filename)`: Tự động sinh tên file processed từ raw.

### `mlflow_tracking_config.py`

- **`MLflowTrackingConfig`**: Cấu hình MLflow.
  - `enabled`: Bật/tắt tracking.
  - `experiment_name`: Tên experiment (mặc định: `"dataset_pipeline"`).
  - `track_meta_files`: Tự động track file meta.
  - `default_tags`: Tags mặc định cho mỗi run.

---

## 🧹 Dataset Cleaning (`src/dataset/cleaning.py`)

### Text-level cleaning functions

| Function                                       | Mô tả                                        |
| ---------------------------------------------- | -------------------------------------------- |
| `remove_html_and_entities(text)`               | Xóa thẻ HTML + giải mã HTML entities         |
| `remove_urls(text)`                            | Xóa URL (http, https, ftp, www)              |
| `remove_mentions(text)`                        | Xóa @username                                |
| `remove_emoji(text)`                           | Xóa emoji                                    |
| `remove_special_chars(text, keep_punctuation)` | Giữ chữ, số, khoảng trắng + dấu câu chỉ định |

### Dataset-level cleaning functions

| Function                        | Mô tả                               | Điểm quan trọng                                                                       |
| ------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------- |
| `remove_null_or_empty(df)`      | Xóa null/empty comment + null label | **Có threshold**: nếu null label > 5% thì raise ValueError                            |
| `remove_non_text_comments(df)`  | Xóa comment không có chữ cái        | Dùng regex `[A-Za-zÀ-ỹ]`                                                              |
| `remove_duplicate_comments(df)` | Xử lý duplicate                     | **Cùng comment + cùng label** → giữ đầu; **cùng comment + khác label** → xóa cả group |
| `remove_outliers(df)`           | Isolation Forest                    | Tự động chọn feature cols numeric nếu không chỉ định                                  |

---

## 📊 EDA (`src/dataset/eda.py`)

### Visualization functions

| Function                         | Mô tả                                          |
| -------------------------------- | ---------------------------------------------- |
| `plot_label_distribution()`      | Biểu đồ phân bố nhãn                           |
| `plot_length_distribution()`     | Histogram char_len + word_len                  |
| `plot_length_boxplot_by_label()` | Boxplot so sánh độ dài giữa các lớp            |
| `plot_ttr_boxplot_by_label()`    | Boxplot TTR giữa các lớp                       |
| `plot_wordcloud()`               | Word cloud từ tokens                           |
| `compare_pos_tags()`             | So sánh POS tags giữa 2 lớp (dùng underthesea) |

### Analysis functions

| Function                                 | Mô tả                                              |
| ---------------------------------------- | -------------------------------------------------- |
| `detect_outliers_isolation_forest()`     | Phát hiện outlier + vẽ biểu đồ                     |
| `preprocess_text_for_eda()`              | Chuẩn hóa + tokenize + filter stopwords cho EDA    |
| `get_top_tokens_by_document_frequency()` | Token xuất hiện nhiều nhất theo document frequency |
| `get_top_ngrams()`                       | Top n-grams phổ biến                               |
| `get_top_ngrams_by_label()`              | Top n-grams riêng cho từng lớp                     |
| `get_ratio_features()`                   | Từ đặc trưng theo tỷ lệ tần suất giữa 2 lớp        |

---

## 🔧 Feature Engineering (`src/dataset/feature_engineering.py`)

| Function                             | Mô tả                                                            |
| ------------------------------------ | ---------------------------------------------------------------- |
| `add_length_features(df)`            | Thêm `char_len`, `word_len`                                      |
| `add_punctuation_emoji_features(df)` | Thêm `num_exclamation`, `num_question`, `num_upper`, `num_emoji` |
| `add_tokens_column(df)`              | Thêm cột `tokens` (chỉ tokenize, không norm/clean/stopwords)     |
| `add_ttr_column(df)`                 | Thêm cột `ttr` (Type-Token Ratio)                                |

---

## ⚙️ Preprocessing (`src/dataset/preprocessing.py`)

| Function                   | Mô tả                                 | Điểm quan trọng                               |
| -------------------------- | ------------------------------------- | --------------------------------------------- |
| `get_tokenizer()`          | Lazy load underthesea word_tokenize   | Có thể override                               |
| `get_stopwords()`          | Lazy load stopwordsiso cho tiếng Việt | Có thể override                               |
| `normalize_text(text)`     | NFC normalize + lower + strip spaces  | Dùng `unicodedata.normalize("NFC", ...)`      |
| `filter_stopwords(tokens)` | Lọc stopwords                         | `return_tokens=True` → list, `False` → string |

---

## ✂️ Split (`src/dataset/split.py`)

| Function            | Mô tả                          | Điểm quan trọng                                                    |
| ------------------- | ------------------------------ | ------------------------------------------------------------------ |
| `split_indices()`   | Core: chia indices thành 3 tập | Dùng `train_test_split` 2 lần. Kiểm tra tổng tỷ lệ = 1.0           |
| `split_dataframe()` | Wrapper: chia DataFrame        | `return_indices=True` → indices, `False` → DataFrame (reset_index) |

---

## ✅ Validation (`src/dataset/validation.py`)

| Function                       | Mô tả                             | Điểm quan trọng                                                                                   |
| ------------------------------ | --------------------------------- | ------------------------------------------------------------------------------------------------- |
| `check_null(df)`               | Kiểm tra null/NaN                 | Trả về null_counts, total_null, null_ratio                                                        |
| `check_empty_or_no_letter(df)` | Kiểm tra rỗng + không chữ cái     | Có thể bật/tắt từng check                                                                         |
| `check_duplicates(df)`         | Phát hiện duplicate + mixed-label | **Quan trọng**: normalize text trước khi so sánh, phát hiện mixed-label (cùng comment khác label) |
| `check_column_names(df)`       | Kiểm tra cột bắt buộc             | Trả về dict `{col: bool}`                                                                         |

---

## 🔄 Pipeline System (`src/pipeline/`)

### `base_pipeline.py`

- **`PipelineStep`**: Dataclass gồm `name`, `enabled_flag`, `func`, `kwargs`.
- **`BasePipeline.run_steps(data, config, steps)`**: Static method chạy các step theo thứ tự. Chỉ chạy step được bật (enabled_flag = True trong config). Trả về `(data, report_dict)`.

### `cleaning_pipeline.py`

- **`CLEANING_STEPS`**: 10 steps theo thứ tự:
  1. Text-level: html → urls → mentions → emoji → special_chars
  2. Dataset-level: null/empty → non-text → normalize_before_dedup → duplicates → outliers
- **`clean_text_pipeline(df)`**: Chạy pipeline + tự động lưu `cleaning_log.json` vào `meta/`.

### `preprocessing_pipeline.py`

- **`PREPROCESSING_STEPS`**: normalize → tokenize → filter_stopwords.
- **`preprocess_text_pipeline(text)`**: Chạy pipeline + lưu `preprocessing_params.json`.
- **`get_tokenizer_from_config()`**, **`get_stopwords_from_config()`**: Helper lấy tokenizer/stopwords từ config.

### `validation_pipeline.py`

- **`VALIDATION_STEPS`**: column_check → null_check → empty_or_no_letter_check → duplicate_check.
- **`validate_dataset(df)`**: Chạy pipeline + lưu `validation_report.json`.

---

## 📡 Tracking (`src/tracking/`)

### `mlflow_tracker.py`

- **`MLflowTracker`**: Context manager cho MLflow.
  - `__enter__`: Set tracking URI, experiment, start run.
  - `__exit__`: End run.
  - Methods: `log_params()`, `log_metrics()`, `log_artifact()`, `log_dict_as_yaml()`, `log_dict_as_json()`, `log_model()`, `set_tag()`.
  - `log_model()`: Tự động chọn sklearn hay pyfunc.

### `dataset_tracker.py`

- **`get_dataset_hash(dataset_path)`**: Tính MD5 hash của thư mục dataset (8 ký tự đầu).
- **`track_dataset_meta(tracker, meta_dir)`**: Đọc tất cả file trong `meta/` và log vào MLflow.
  - File YAML/JSON → parse → extract metrics (số) + params (không số) → log.
  - File không parse được → log artifact gốc.
- **`mlflow_run_exists()`**: Kiểm tra run đã tồn tại dựa trên dataset_hash tag.

### `experiment_manager.py`

- **`ExperimentManager`**: Quản lý experiment lifecycle.
  - `list_experiments()`, `list_runs()`: Liệt kê.
  - `compare_runs()`: So sánh nhiều runs.
  - `get_best_run(metric, mode)`: Tìm run tốt nhất.
  - `export_comparison_html()`: Export bảng so sánh ra HTML.

---

## 🎯 Training Evaluation (`src/training/`)

Module đánh giá mô hình dùng chung cho mọi experiment.
Được trích xuất từ code lặp trong 15 notebooks training (Bản 3 → Bản 17).

### `threshold.py`

| Function                                                                 | Mô tả                                                  | Điểm quan trọng                                       |
| ------------------------------------------------------------------------ | ------------------------------------------------------ | ----------------------------------------------------- |
| `find_best_threshold(y_true, y_prob, metric)`                            | Tìm threshold tối ưu theo metric (f1/precision/recall) | Mặc định thử 91 thresholds từ 0.05 → 0.95             |
| `find_best_threshold_cost(y_true, y_prob, cost_fp, cost_fn)`             | Tìm threshold tối ưu dựa trên chi phí FP/FN            | Total cost = FP*cost_fp + FN*cost_fn                  |
| `find_best_threshold_with_recall_constraint(y_true, y_prob, min_recall)` | Tìm threshold với ràng buộc recall >= min_recall       | Trong các threshold thoả mãn, chọn precision cao nhất |

### `metrics.py`

| Function                                      | Mô tả                                                 | Điểm quan trọng                                                      |
| --------------------------------------------- | ----------------------------------------------------- | -------------------------------------------------------------------- |
| `evaluate_model(model, X, y_true, threshold)` | Predict + in classification report + confusion matrix | Trả về dict đầy đủ metrics + y_pred + y_prob                         |
| `run_cross_validation(model, X, y, cv)`       | Chạy cross-validation với StratifiedKFold             | Mặc định 6 metrics: accuracy, precision, recall, f1, roc_auc, pr_auc |
| `print_cv_results_table(cv_scores)`           | Chuyển CV scores thành DataFrame                      | Cột: metric, mean, std, min, max                                     |

### `visualization.py`

| Function                                 | Mô tả                                     |
| ---------------------------------------- | ----------------------------------------- |
| `plot_roc_curve(y_true, y_prob)`         | Vẽ ROC curve                              |
| `plot_pr_curve(y_true, y_prob)`          | Vẽ Precision-Recall curve                 |
| `plot_calibration_curve(y_true, y_prob)` | Vẽ Calibration curve (dùng quantile bins) |

### `error_analysis.py`

| Function                                                  | Mô tả                                            |
| --------------------------------------------------------- | ------------------------------------------------ |
| `show_fp_fn_samples(comments, y_true, y_pred, y_prob, n)` | Hiển thị FP/FN samples, sắp xếp theo probability |

---

## 🛠️ Utils (`src/utils/`)

### `csv.py`

- **`read_csv_with_columns(data_path)`**: Đọc CSV + kiểm tra cột bắt buộc. Nếu thiếu cột → `SystemExit`.

### `downloader.py`

- **`download(folder_url, dest_dir, filename=None)`**: Tải từ Google Drive bằng gdown.
  - `filename=None`: Tải toàn bộ folder.
  - `filename=...`: Tải 1 file cụ thể, tự động tìm trong subdirectory và move lên đúng thư mục.

---

## 📝 Notebooks Training

### Bản 3: Baseline (LogisticRegression + TF-IDF)

- **Pipeline**: Preprocess cơ bản (fillna, ép string) → TF-IDF (ngram_range=(1,2), max_df=0.9, min_df=2) → LogisticRegression (class_weight='balanced')
- **Đặc điểm**: Baseline thuần túy, chưa xử lý ngôn ngữ tiếng Việt (không tokenize, không stopwords, không teen code, không emoji)
- **Split**: 70/15/15, stratify
- **Evaluation**: ROC, PR, calibration, CV 5-fold, FP/FN

### Bản 4: LogisticRegression + GridSearch (có preprocessing tiếng Việt)

- **Cải tiến so với bản 3**: Thêm tokenize bằng underthesea, thay thế teen code, xóa stopwords tiếng Việt (bộ thủ công), giữ lại dấu !?
- **Pipeline**: Preprocessing nâng cao → TF-IDF (trong pipeline) → GridSearchCV (ngram_range, max_features, min_df, max_df, C) → LogisticRegression (class_weight='balanced')
- **Hạn chế**: Chưa có đặc trưng số học, chưa xử lý emoji

### Bản 5: LogisticRegression + đặc trưng số + emoji

- **Cải tiến so với bản 4**: Thêm chuyển emoji → text, thêm đặc trưng số học (độ dài, số !, số ?, số chữ hoa, số emoji) từ comment gốc → dùng `hstack` ghép với TF-IDF
- **Pipeline**: Preprocessing (tokenize, teen code, stopwords, emoji) → TF-IDF + đặc trưng số → LogisticRegression (GridSearch tìm C + class_weight)
- **Đặc điểm**: Phiên bản tiến bộ nhất trong nhóm LogisticRegression: xử lý tiếng Việt + đặc trưng số + emoji + tự động chọn class_weight tối ưu

### Bản 6: FastText (mean+max pooling) + LogisticRegression

- **Pipeline**: Preprocess cơ bản (NFC, lower, remove special chars, tokenize) → FastText pre-trained 300 chiều → mean+max concatenate (600 chiều) → LogisticRegression (class_weight='balanced')
- **Đặc điểm**: Phiên bản FastText đơn giản nhất, chưa tối ưu, không chuẩn hóa, không GridSearch
- **Hạn chế**: Không xử lý teen code, không dấu câu, không emoji, không stopwords

### Bản 7: FastText + đặc trưng số + L2 normalization

- **Cải tiến so với bản 6**: Thêm 5 đặc trưng số học (độ dài, số !, số ?, số chữ hoa, số emoji) → hstack (605 chiều) + L2 normalization toàn bộ vector
- **Pipeline**: FastText mean+max pooling (600) + đặc trưng số (5) → L2 normalize → LogisticRegression (class_weight='balanced')
- **Split**: 70/15/15, stratify, dùng indices để lưu comment gốc hiển thị FP/FN

### Bản 8: FastText + preprocessing nâng cao (giữ dấu câu, teen code)

- **Cải tiến so với bản 7**: Preprocessing nâng cao: giữ lại dấu câu (!?.,:;) và thêm khoảng trắng bao quanh để thành token riêng, mở rộng teen code (~30 từ lóng)
- **Pipeline**: Giống bản 7 (FastText 600 + 5 features → L2 normalize → LogisticRegression)
- **Kết quả**: Không cải thiện so với bản 7

### Bản 9: FastText + TF-IDF weighted average (cải tiến quan trọng)

- **Cải tiến so với bản 8**: Thay mean+max pooling bằng **weighted average với TF-IDF**: mỗi từ trong câu được nhân với trọng số TF-IDF toàn cục trước khi lấy trung bình
- **Pipeline**: FastText 300 chiều + TF-IDF weighted average + 5 đặc trưng số → 305 chiều → L2 normalize → LogisticRegression
- **Kết quả**: Có cải thiện rõ so với bản 8

### Bản 10: FastText fine-tuned + TF-IDF weighted average (overfit)

- **Cải tiến so với bản 9**: Fine-tune FastText unsupervised (skip-gram, 5 epochs) trên tập train (4446 câu) bắt đầu từ pre-trained
- **Pipeline**: FastText fine-tuned → TF-IDF weighted average + 5 features → L2 normalize → LogisticRegression
- **Kết quả**: **Overfit nghiêm trọng** - recall rất cao (93.3%) nhưng precision cực thấp (67.9%), không dùng được thực tế

### Bản 11: FastText + TF-IDF weighted average (cải tiến bản 10, lần 1)

- **Pipeline**: Preprocess basic (NFC + lower + remove punctuation + word_tokenize) → FastText pre-trained (không fine-tune) → TF-IDF weighted average + 5 features → L2 normalize → LogisticRegression (class_weight='balanced')
- **Threshold**: Cost-based (FP=1, FN=5) → best threshold ≈ 0.2432
- **Kết quả test**: Precision ≈ 0.86, Recall ≈ 0.76, F1 ≈ 0.81, PR-AUC ≈ 0.93
- **Đặc điểm**: Quay lại dùng pre-trained (bỏ fine-tune), dùng cost-based threshold

### Bản 12: FastText + TF-IDF weighted average (cải tiến bản 10, lần 2)

- **Pipeline**: Giống bản 11 (preprocess basic → FastText pre-trained → TF-IDF weighted average + 5 features → L2 normalize → LogisticRegression)
- **Threshold**: Cost-based + fixed thresholds [0.30, 0.32, 0.35, 0.38], chọn threshold có recall ≥ 0.95 và precision cao nhất → best threshold ≈ 0.3210
- **Kết quả test**: Precision ≈ 0.86, Recall ≈ 0.76, F1 ≈ 0.81, PR-AUC ≈ 0.93
- **Đặc điểm**: Khác bản 11 ở cách chọn threshold (ưu tiên recall ≥ 95%)

### Bản 13: FastText + LightGBM (baseline)

- **Pipeline**: Preprocess basic (NFC + lower + remove punctuation + word_tokenize) → TF-IDF weighted FastText vectors → LightGBM (class_weight='balanced', n_estimators=200, lr=0.05, max_depth=5)
- **Threshold selection**: Cost-based (FP cost 1, FN cost 5) + fixed thresholds [0.30-0.42]
- **Evaluation**: ROC-AUC, PR-AUC, Calibration curve, 5-fold CV, FP/FN analysis

### Bản 14: FastText + LightGBM (improved)

- **Cải tiến**: GridSearchCV (n_estimators, learning_rate, max_depth, num_leaves, reg_alpha, reg_lambda) với 3-fold CV, scoring='f1'
- **Threshold**: Chọn theo F1 max (không ràng buộc recall)

### Bản 15: FastText + CatBoost

- **Model**: CatBoostClassifier với auto_class_weights='Balanced'
- **GridSearch**: iterations, learning_rate, depth, l2_leaf_reg, border_count

### Bản 16: PhoBert + LightGBM

- **Feature**: Sử dụng PhoBert embeddings thay vì FastText
- **Model**: LightGBM trên PhoBert vectors

### Bản 17: PhoBert + LightGBM + PCA (improved)

- **Cải tiến so với bản 16**: Thêm PCA giảm chiều PhoBert embeddings từ 768 → 200 (giữ ~91% variance) + GridSearchCV (n_estimators, learning_rate, max_depth, num_leaves, reg_alpha, reg_lambda) với 3-fold CV, scoring='f1'
- **Pipeline**: Preprocess basic → PhoBert embeddings (768 chiều) → PCA (200 chiều) → LightGBM (best params: lr=0.1, max_depth=7, n_estimators=200, num_leaves=31, reg_alpha=0.1, reg_lambda=0.1)
- **Kết quả**: Best CV F1 ≈ 0.8041, test accuracy ≈ 0.7737
- **Đặc điểm**: Dùng PCA để giảm chiều PhoBert embeddings, GridSearch tìm tham số tối ưu cho LightGBM

## 🔑 Key Design Patterns & Principles

1. **Config-driven**: Mọi tham số đều qua config, có thể override bằng `override()`.
2. **Pipeline pattern**: `BasePipeline.run_steps()` với `PipelineStep` dataclass.
3. **Lazy loading**: Tokenizer và stopwords được load khi cần, có thể override.
4. **Separation of concerns**: Cleaning, preprocessing, feature engineering, EDA, validation được tách riêng.
5. **Reproducibility**: Random state cố định, stratified split, logging đầy đủ.
6. **Meta tracking**: Tự động lưu JSON report vào `meta/` + MLflow tracking.
7. **Pure functions**: Các hàm trong `src/dataset/` là pure, không phụ thuộc config.
8. **Wrapper pattern**: Pipeline steps dùng wrapper functions để chuẩn hóa interface `(data, **kwargs) -> (data, dict)`.

---

## ⚠️ Important Notes & Gotchas

- **cleaning.py `remove_null_or_empty`**: Nếu null label ratio > 5% → raise ValueError (dừng pipeline).
- **cleaning.py `remove_duplicate_comments`**: Cùng comment + khác label → xóa **toàn bộ group** (không giữ lại).
- **validation.py `check_duplicates`**: Có thể normalize text trước khi so sánh. Phát hiện mixed-label.
- **split.py**: Kiểm tra tổng tỷ lệ phải = 1.0 (dùng `abs(sum - 1.0) < 1e-9`).
- **feature_engineering.py `add_tokens_column`**: Chỉ tokenize, KHÔNG normalize/clean/filter.
- **preprocessing.py**: Dùng `unicodedata.normalize("NFC", ...)` cho tiếng Việt.
- **downloader.py**: `gdown.download_folder` có thể tạo thư mục con → tự động tìm và move file.
- **path_config.py `get_processed_filename()`**: Tự động bỏ prefix "raw*" và thêm "processed*".
- **dataset_tracker.py**: Metrics và params được prefix bằng tên file (VD: `cleaning_log.final_rows`).
