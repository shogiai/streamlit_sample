# 必要なパッケージをインポート
import streamlit as st # type: ignore
import pandas as pd # type: ignore
import requests # type: ignore
from streamlit_option_menu import option_menu # type: ignore

# 追加パッケージの導入が必要となる
# $ pip install streamlit-option-menu

# ページコンフィグ
title = 'Day 29 - Zero-Shot Text Classifier'
st.set_page_config(layout='wide', page_title=title, page_icon='🐻')

# タイトルを2カラムレイアウトで配置
c1, c2 = st.columns([0.3, 2])
# １カラム目にロゴを表示
with c1:
    # オリジナルソースコードでは画像はここからダウンロードして、このファイルと同じディレクトリに置く
    # https://github.com/CharlyWargnier/zero-shot-classifier/blob/main/logo.png
    st.image('img/logo.png', width=110)
# 2カラム目にタイトルを表示
with c2:
    st.title(title)

# サイドバーにメニューを表示
with st.sidebar:
    selected_model = option_menu(
        '',
        ['Japanese Model', 'Original Model'],
        icons=['globe-asia-australia', 'globe-americas'],
        menu_icon='',
        default_index=0,
    )

# Hugging Face API Token
# ハードコーディングするのは危険なので、Secrets managementを使う
# https://docs.streamlit.io/library/advanced-features/secrets-management
# ローカルで実行する場合は、.streamlit/secrets.tomlにAPI_TOKENを記載する
api_token = st.secrets['API_TOKEN']
headers = {'Authorization': 'Bearer {}'.format(api_token)}

# APIのURLを選択
api_url_japanese = 'https://api-inference.huggingface.co/models/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
api_url_original = 'https://api-inference.huggingface.co/models/valhalla/distilbart-mnli-12-3'
if selected_model == 'Japanese Model':
    api_url = api_url_japanese
else:
    api_url = api_url_original

# Hugging Face APIを呼び出す関数。Hugging Faceのサイトで生成されるサンプルコードそのまま
def query(payload):
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()

# サンプルの文章
sample1 = '国の特別天然記念物ライチョウを長野県の中央アルプスへ返す取り組みを続けている那須どうぶつ王国。先月8月に生まれたライチョウのヒナ3羽がそろって生後1カ月を迎えた。'
sample2 = 'アルファベット傘下のグーグルは30日、日本とインドで検索サービスに生成AIを導入した。ユーザーが入力した質問や指示に対して要約などを含むテキストやビジュアルで結果を表示することができる。日本では日本語、インドでは英語とヒンディー語に対応する。'
sample3 = '毎年12月10日に開かれるノーベル賞の授賞式や晩餐会について、ノーベル財団が今年はロシアやベラルーシの駐在大使も招待すると発表した。ウクライナ侵攻を受け、昨年は招待を見送っていた。'
sample4 = 'Ancient Greece was a crucial culture that laid the foundation for Western civilization. They gave birth to democracy and produced philosophers (Socrates, Plato, Aristotle), poets (Homer), mathematicians (Pythagoras), and initiated the Olympic Games. Greek mythology with gods like Zeus and heroes like Hercules is famous. Their artistic, architectural, and scientific legacy continues to influence the modern world.'
sample5 = 'Cats are small mammals with the following biological features. They belong to the Felidae family and are primarily carnivorous, eating herbivorous animals. Cats have sleek bodies with agile muscles and excellent claws, well-suited for hunting. They are nocturnal, exhibiting great night vision. Cats possess fluffy tails and distinctive whiskers, aiding in balance and sensory perception.'

# 文章を入力
st.subheader('ラベル分類したい文章')
selected_sequence = st.selectbox('サンプル文章を選択', ['', sample1, sample2, sample3, sample4, sample5])
sequence = st.text_area('文章', value=selected_sequence, height=200)

# ラベルを入力
st.subheader('ラベル定義')
label1 = st.text_input('ラベル1', value='政治')
label2 = st.text_input('ラベル2', value='テクノロジー')
label3 = st.text_input('ラベル3', value='動物')
# 空文字でないものをリストにする
labels = [l for l in [label1, label2, label3] if l != '']

# 分類開始
if st.button('分類する'):
    # APIを呼び出す
    payload = {
        'inputs': sequence,
        'parameters': {'candidate_labels': labels},
    }
    output = query(payload)

    # 出力をパースして表示
    df = pd.DataFrame()
    df['labels'] = output['labels']
    df['scores'] = output['scores']
    st.dataframe(df)

    # デバッグ表示
    with st.expander('Hugging Face API Response'):
        st.write(output)
