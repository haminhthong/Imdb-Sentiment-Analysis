# Dataset layout

`data/raw/` chứa Large Movie Review Dataset official sau khi chạy
`scripts/download_imdb.py`. `data/processed/` dành cho dữ liệu đã chuẩn hóa.
Hai thư mục này được gitignore vì dữ liệu đầy đủ không thuộc source repository.

Fixture smoke nằm ở `tests/fixtures/imdb_smoke/` và chỉ dùng cho test/CI; nó
không được đặt tên `train.csv`/`test.csv` ở root và không đại diện cho IMDB.
