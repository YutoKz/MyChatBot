# MyChatBot
### Streamlit Cloud
[URL](https://mychatbot-05.streamlit.app/)
- 時間割やシラバスは架空のもの
- シラバスはPDF形式でアップロード
- 口コミ機能
- シラバスや口コミはOpenAIのAPIを利用してベクトルに埋め込み。ベクトルDBであるQdrantに保管。
- RAGの実装
  - ユーザからシラバスや口コミに関する質問を受け付ける
  - 質問文と類似する文章をベクトルDBから探索、上位10個をプロンプトに埋め込む
  - OpenAIのAPIを利用してChatGPTに入力、出力を表示
- [使い方のイメージ]
  - ユーザは自身の所属を選択することで時間割を参照
  - シラバスや口コミについてLLMに質問・相談しながら、履修する選択科目を決定
  - 長く読みづらいシラバスや、科目毎の口コミを調べる手間を省くことで、効率的に時間割を組むことが可能

### 目標
[参考](https://zenn.dev/ml_bear/books/d1f060a3f166a5/viewer/f11592)を実装し、<br>
発展させて自作のChatBotを作りたい

### アイデア
- シラバスをデータベースとして保持しておき、相談しながら一緒に時間割を組んでくれる

### 環境構築
パッケージ管理ツールとして[`Rye`](https://rye.astral.sh/)を使用
##### pyproject.toml
OpenAI_API_KEY が記載されているファイルは GitHub にプッシュできないので、テンプレートを `pyproject_template.txt` に記載



