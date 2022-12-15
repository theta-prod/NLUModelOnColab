# [REF] GPT2-內容擴增

- **Metadata** : `type: REF` `scope: NLP, GPT2` 
- **Techs Need** : `transformer` `gpt`
- **Status**: `need-review`
<br/><br/>

## ✨ You should already know
- huggingface
- transformer
- gpt
- autoModel, pipelines

👩‍💻 👨‍💻

## ✨ About the wiki
- `Situation:` 我們在編寫內容文件時，希望有範例可以做為參考來修正。
- `Target:`根據部分內容來產生更多的內容來做為範例
- `Index:`

| Sub title | decription | memo |
| ------ | ------ | ------ |
| Data | 使用公用資料庫來準備訓練資料 | 包含下載與資料前處理 |
| Model | 透過huggingface的Trainer來訓練 | 包含創建訓練資料集與使用訓練模型 |
| Infer | 透過huggingface的pipeline來應用 | 包含應用方式與功能展示 |


---
<br>

### **Data**
> 使用公用資料庫來準備訓練資料
####  📝 下載資料
> 首先你必須先有帳號與token，然後使用API下載

- 安裝與登入
```
!pip install kaggle
!mkdir ~/.kaggle
!touch ~/.kaggle/kaggle.json
with open("/root/.kaggle/kaggle.json", "w") as f:
  f.write('{"username":"","key":""}')

!chmod 600 /root/.kaggle/kaggle.json

```


- 下載資料中文摘要資料
```

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
api.dataset_download_files("terrychanorg/chinese-simplified-xlsum-v2", path="./", unzip=True)
```

- 讀取中文繁體資料
```
import re
import json
from sklearn.model_selection import train_test_split

data = []
with open('./chinese_traditional_XLSum_v2.0/chinese_traditional_val.jsonl') as f:
    for line in f.readlines():
      line.replace("\n","")
      data.append(json.loads(line))
```

> 每一筆資料會如同下列範例

```
{
	"id":  "zhongwen_trad.141113_asean_myanmar_obama"
	"url":  "https://www.bbc.com/zhongwen/trad/world/2014/11/141113_asean_myanmar_obama"
	"title":  "東亞峰會：奧巴馬警告緬甸改革「倒退」"
	"summary":  "正在緬甸首都內比都出席東南亞國家聯盟系列會議的美國總統奧巴馬在與緬甸總統吳登盛會晤前警告稱，吳登盛政府的改革進程出現倒退。"
	"text":  "一些人批評吳登盛（右）的政府雖然取代了軍閥統治但改革乏力。 以泰國為基地的流亡緬甸人網絡雜誌《伊洛瓦底》刊登了對奧巴馬的書面採訪。奧巴馬說，緬甸的政治改革曾有所進展，但步伐正在放慢，甚至出現了一些倒退。 奧巴馬將於星期四（11月13日）出席伴隨東盟峰會舉行的東亞峰會，並與吳登盛舉行正式會晤。預料他將向吳登盛重覆這些批評。 緬甸在野黨派領袖昂山素季最近也警告說，緬甸的改革進程正面臨停滯不前。 BBC駐仰光記者費舍爾分析說，昂山素季最為不滿的是緬甸仍未修改2008年訂立的憲法。該憲法保障了緬甸軍方在政府去軍事化後的政治生活。 記者說，這部憲法保證軍方仍可取得聯邦議會內四分之一的議席，並對憲法的任何修訂有否決權。 費舍爾指出，憲法的「工程師們」自豪地形容這是「有紀律的民主」。 緬甸將於2015年舉行新一屆大選，但目前的憲法有特定條文禁止昂山素季參選總統。 「不符人民期盼」 奧巴馬 在《伊洛瓦底》的書面採訪中寫道：「緬甸在更新於和解的漫長且艱辛的道路上仍處於起步階段。」 他指出，自2010年11月軍政府開始向文人政府過度依賴，在釋放政治犯、推動憲政改革和與少數民族訂立停戰協定等方面確實取得了進展。 但是，奧巴馬認為，改革的速度並未符合多數人的期盼。 奧巴馬舉例說，一些政治犯從監獄獲釋後仍然受到各種限制；一些新聞記者遭到逮捕；若開邦仍然有羅興亞人因反穆斯林暴力浪潮而流離失所。 奧巴馬將於星期五（14日）與昂山素季會晤。奧巴馬在採訪中寫道：「我尤其有興趣聆聽他對於憲政改革進程的看法，以至於明年的選舉，以及包括美國在內的國際社群能幫助確保這是一場包容、透明和可信的投票。」 這是奧巴馬第二次訪問緬甸，而他在整整一年前的首次訪問也是美國總統歷來首次到訪緬甸。 這次與他同時到訪緬甸的還有中國總理李克強、日本首相安倍晉三、韓國總統朴槿惠、俄羅斯總理梅德韋傑夫、印度總理莫迪和聯合國秘書長潘基文等。 （撰稿：葉靖斯 責編：蕭爾） 如果您對這篇報道有任何意見或感想，歡迎電郵至 chinese@bbc.co.uk。您也可以使用下表給我們發來您的意見："
}
```

####  📝 處理資料


- 訓練資料處理

> 將json轉為內容擴增的訓練資料，這邊我使用了一個特殊關鍵字(`BEG;END`, 個人自訂可隨意替換)來做為摘要與內容的區隔。
> 符合該模型目標，當使用者輸入短句時，將此短句作為主幹進行擴寫。

```
def build_text_files(data_json, dest_path):
    with open(dest_path, 'w') as f:
      data = []
      for texts in data_json:
          title = str(texts['title']).strip()
          text = str(texts['text']).strip()
          summary = str(texts['summary']).strip()
          data.append(f"{summary}BEG;END{text}")
          data.append(f"{title}BEG;END{text}")
      f.write("\n".join(data))

```
> 處理完的資料如下

```
東亞峰會：奧巴馬警告緬甸改革「倒退」BEG;END一些人批評吳登盛（右）的政府雖然取代了軍閥統治但改革乏力。 以泰國為基地的流亡緬甸人網絡雜誌《伊洛瓦底》刊登了對奧巴馬的書面採訪。奧巴馬說，緬甸的政治改革曾有所進展，但步伐正在放慢，甚至出現了一些倒退。 奧巴馬將於星期四（11月13日）出席伴隨東盟峰會舉行的東亞峰會，並與吳登盛舉行正式會晤。預料他將向吳登盛重覆這些批評。 緬甸在野黨派領袖昂山素季最近也警告說，緬甸的改革進程正面臨停滯不前。 BBC駐仰光記者費舍爾分析說，昂山素季最為不滿的是緬甸仍未修改2008年訂立的憲法。該憲法保障了緬甸軍方在政府去軍事化後的政治生活。 記者說，這部憲法保證軍方仍可取得聯邦議會內四分之一的議席，並對憲法的任何修訂有否決權。 費舍爾指出，憲法的「工程師們」自豪地形容這是「有紀律的民主」。 緬甸將於2015年舉行新一屆大選，但目前的憲法有特定條文禁止昂山素季參選總統。 「不符人民期盼」 奧巴馬 在《伊洛瓦底》的書面採訪中寫道：「緬甸在更新於和解的漫長且艱辛的道路上仍處於起步階段。」 他指出，自2010年11月軍政府開始向文人政府過度依賴，在釋放政治犯、推動憲政改革和與少數民族訂立停戰協定等方面確實取得了進展。 但是，奧巴馬認為，改革的速度並未符合多數人的期盼。 奧巴馬舉例說，一些政治犯從監獄獲釋後仍然受到各種限制；一些新聞記者遭到逮捕；若開邦仍然有羅興亞人因反穆斯林暴力浪潮而流離失所。 奧巴馬將於星期五（14日）與昂山素季會晤。奧巴馬在採訪中寫道：「我尤其有興趣聆聽他對於憲政改革進程的看法，以至於明年的選舉，以及包括美國在內的國際社群能幫助確保這是一場包容、透明和可信的投票。」 這是奧巴馬第二次訪問緬甸，而他在整整一年前的首次訪問也是美國總統歷來首次到訪緬甸。 這次與他同時到訪緬甸的還有中國總理李克強、日本首相安倍晉三、韓國總統朴槿惠、俄羅斯總理梅德韋傑夫、印度總理莫迪和聯合國秘書長潘基文等。 （撰稿：葉靖斯 責編：蕭爾） 如果您對這篇報道有任何意見或感想，歡迎電郵至 chinese@bbc.co.uk。您也可以使用下表給我們發來您的意見：
```


- 切割訓練資料並編寫成訓練與測試檔案
```
train, test = train_test_split(data,test_size=0.15)
build_text_files(train,'train_dataset.txt')
build_text_files(test,'test_dataset.txt')
```



### **Model**
> 透過huggingface的Trainer來訓練

####  📝 客製化資料集(`Dataset`)物件
> 這邊由於我們希望可以一行一行的載入資料，因此複製`TextDataset`的原始程式碼來客製化。
```
from torch.utils.data import Dataset
from typing import Any, Optional
import torch
import os
import logging
from filelock import FileLock
import time
import pickle
logger = logging.getLogger()
class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(
        self,
        tokenizer: Any,
        file_path: str,
        block_size: int,
        overwrite_cache=False,
        cache_dir: Optional[str] = None,
    ):

        if os.path.isfile(file_path) is False:
            raise ValueError(f"Input file path {file_path} not found")

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else directory,
            f"cached_lm_{tokenizer.__class__.__name__}_{block_size}_{filename}",
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    texts = f.readlines()


                for text in texts:
                  tokenedtext = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
                  for i in range(0, len(tokenedtext) - block_size + 1, block_size):
                    self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenedtext[i : i + block_size]))
                
                
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should look for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    f"Saving features into cached file {cached_features_file} [took {time.time() - start:.3f} s]"
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)
```

####  📝 選擇模型
模型這邊我們有用這兩個測試，效果都相當不錯
```
MODEL_NAME = "uer/gpt2-chinese-cluecorpussmall"
# MODEL_NAME = "ckiplab/gpt2-base-chinese"
```

- 載入transformers
接下來我們直接進行載入動作，這邊我們為了讓該流程具有可複製性，因此我們設定了亂數子。
```
!pip install transformers>=4.2.2

from transformers import set_seed, AutoTokenizer
set_seed(123)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
```


- 製作訓練資料集與測試資料集
```
from transformers import DataCollatorForLanguageModeling

def load_dataset(train_path,test_path,tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=512)
     
    test_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=test_path,
          block_size=512)   
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset,test_dataset,data_collator

train_path = 'train_dataset.txt'
test_path = 'test_dataset.txt'
train_dataset,test_dataset,data_collator = load_dataset(train_path,test_path,tokenizer)

```

> 成功的話會像這樣

```
train_dataset[:3]
     
>>> tensor([
		[ 101, 5401, 1751,  ..., 4294, 1762,  102],
        [ 101, 5401, 1751,  ..., 4525, 3280,  102],
        [ 101, 2111, 2094,  ...,  889, 6303,  102]
	])
```



####  📝 訓練參數

訓練參數如下：

```
from transformers import Trainer, TrainingArguments,AutoModelWithLMHead

model = AutoModelWithLMHead.from_pretrained(MODEL_NAME)


training_args = TrainingArguments(
    output_dir="./gpt2-reporter", #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=2, # number of training epochs
    per_device_train_batch_size=8, # batch size for training
    per_device_eval_batch_size=8,  # batch size for evaluation
    evaluation_strategy="steps",
    eval_steps = 400, # Number of update steps between two evaluations.
    logging_steps= 400,
    save_steps = 800, # after # steps model is saved 
    warmup_steps=500,# number of warmup steps for learning rate scheduler
    prediction_loss_only=True,
    load_best_model_at_end=True,
    save_total_limit=3,
    learning_rate=5e-5,
    push_to_hub=True,
    hub_model_id="theta/gpt2-reporter"
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
```

> 開始訓練
```
trainer.train()
```

### **Infer**
> 透過huggingface的pipeline來應用
####  📝 載入模型
```
from transformers import pipeline, AutoModelWithLMHead
model = AutoModelWithLMHead.from_pretrained('theta/gpt2-reporter')
reporter = pipeline('text-generation',model=model, tokenizer=MODEL_NAME)
```

####  📝 使用範例 - 擴增
> 當使用者輸入短句時，將此短句作為主幹進行擴寫。
> 請在短句後面加入`BEG;END`作為特徵點。

```

reporter('總統宣布國防預算大漲BEG;END',max_length=800)

>>> [
{
	'generated_text': 
	'總統宣布國防預算大漲BEG;END 美 國 就 航 母 建 設 提 出 要 求 。 雖 然 美 國 承 諾 暫 停 向 中 國 出 售 這 艘 航 空 母 艦 ， 但 到 目 前 為 止 ， 航 母 開 發 的 總 價 和 投 資 遠 遠 高 於 美 國 的 競 標 基 金 ， 中 國 主 導 的 航 母 項 目 ， 預 計 將 比 總 投 資 上 升 1000 % 左 右 是 否 能 夠 實 現 其 超 重 排 放 目 標 ？ 最 新 新 聞 報 道 稱 ， 由 於 美 國 還 想 繼 續 向 中 國 出 售 現 有 的 航 母 設 施 ， 因 此 新 增 預 算 可 能 超 過 美 國 其 他 預 算 ， 其 中 包 括 對 航 空 母 艦 的 投 資 。 這 其 中 包 括 建 造 更 多 的 新 型 號 航 母 、 改 進 和 更 強 大 的 艦 載 機 系 統 以 及 對 抗 性 巡 航 導 彈 等 。 美 國 主 管 國 防 預 算 的 副 國 務 卿 蓬 佩 奧 表 示 ， 即 使 美 國 無 法 在 本 屆 歐 洲 戰 略 盟 友 日 的 行 動 中 參 與 軍 事 行 動 也 能 夠 解 決 美 國 的 航 母 動 亂 的 問 題 ， 對 於 中 國 來 說 也 極 為 關 鍵 ， 美 國 將 會 面 對 更 多 的 挑 戰 。 外 界 普 遍 預 測 ， 除 非 航 母 出 現 爆 炸 危 機 ， 中 國 可 能 會 大 力 度 投 資 航 母 。 因 此 ， 蓬 佩 奧 提 到 了 大 約 60 萬 美 元 ， 來 向 美 國 「 要 回 金 錢 」 。 蓬 佩 奧 說 ， 由 於 航 母 項 目 的 支 出 將 遠 遠 超 過 美 國 預 算 ， 中 國 自 己 想 發 起 的 軍 事 行 動 能 夠 實 現 其 超 重 排 放 目 標 ， 對 於 美 國 也 極 為 關 鍵 。 蓬 佩 奧 說 ， 外 國 主 管 國 防 預 算 的 副 國 務 卿 蓬 佩 奧 在 周 五 （ 10 月 27 日 ） 上 午 就 軍 演 的 具 體 行 動 承 諾 給 中 國 提 供 了 建 議 ， 美 國 需 要 在 今 後 多 長 時 間 上 對 航 母 動 亂 提 出 更 多 的 抗 議 。 他 還 說 ， 如 果 中 國 真 的 能 發 起 這 場 航 母 動 亂 ， 會 在 未'
}]
```


####  📝 使用範例 - 延展
> 也可以直接用於基本的gpt功能，接續寫內容。
```

reporter('總統宣布國防預算大漲',max_length=800)

>>> [
{
	'generated_text': 
	'總統宣布國防預算大漲 30 億 美 元 ， 美 國 政 府 的 貨 幣 刺 激 政 策 正 進 入 蜜 月 期 。 因 此 ， 儘 管 美 國 對 中 國 實 施 「 財 政 緊 縮 」 以 應 對 經 濟 增 長 放 緩 ， 中 國 是 否 仍 然 採 取 一 定 措 施 、 遏 制 以 中 國 國 利 貸 為 基 礎 實 施 的 貿 易 戰 仍 在 進 行 中 。 貿 易 戰 對 經 濟 的 影 響 ？ 由 於 全 球 經 濟 陷 入 衰 退 ， 中 國 在 2017 年 實 際 對 外 開 放 進 口 總 值 為 328 億 美 元 ， 比 去 年 增 長 了 0. 8 % ， 比 2007 全 年 增 長 2. 9 % ， 但 是 到 了 2019 年 ， 中 國 對 外 開 放 仍 然 以 增 長 3. 5 % 的 速 度 邁 上 3 % 的 台 階 。 一 些 評 論 人 士 認 為 ， 中 國 的 經 濟 增 長 也 進 入 了 「 瓶 頸 」 。 因 為 全 球 經 濟 的 衰 退 ， 中 國 在 去 年 的 增 速 在 7. 6 % 以 下 ， 而 且 今 年 的 經 濟 增 速 也 下 滑 了 3 % 。 中 國 政 府 宣 佈 的 「 財 政 緊 縮 」 政 策 是 根 據 中 國 國 家 統 計 局 公 布 的 經 濟 增 長 指 標 結 果 進 行 計 算 的 。 其 中 主 要 分 析 指 出 ， 中 國 經 濟 的 確 很 差 ， 在 這 種 長 期 惡 化 的 狀 況 下 ， 如 果 要 使 經 濟 繼 續 保 持 「 快 速 復 蘇 」 ， 就 必 須 依 靠 大 量 財 政 來 支 撐 。 中 國 財 政 緊 縮 是 如 何 造 成 的 ？ 財 政 部 周 三 就 《 中 國 共 產 黨 新 聞 工 作 條 例 》 開 展 聯 合 調 查 ， 並 在 官 方 微 博 @ 國 防 觀 察 網 刊 登 新 聞 稿 表 示 ， 雖 然 官 方 的 經 濟 調 查 顯 示 ， 中 國 經 濟 在 國 際 經 濟 復 蘇 衰 退 的 狀 況 下 能 在 2017 年 年 底 達 到 4 % ， 但 是 經 濟 卻 出 現 了 不 同 程 度 回 落 。 分 析 人 士 指 出 ， 中 國 經 濟 正 在 經 歷 類 似 2008 全 年 的 經 濟 增 速 放 慢 ， 不 斷 下 降 的 目'
}]
```