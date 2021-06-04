# atmacup8

主に1サブ1コミットで実験をしています。コミットログから、CVとPublic LBのスコアをみれます。

---

https://www.guruguru.science/competitions/13/


## 概要
>ゲームの情報を使って、ゲームの売上を予測します。使えるデータに関しては data sources から閲覧できます。

## Timeline
> 2020/12/04 18:00 ~ 2020/12/13 18:00 (Asia/Tokyo)

## 評価指標
> root mean squared logarithmic error によって評価します。  
$RMSLE = \sqrt{\frac{1}{n} \sum^n_{i=1} (log(t_i - 1) - log(y_i +1))^2}$

## 取り組み
はてなブログにまとめました。
- [atmaCup#8参加記（pop-ketle版）](https://pop-ketle.hatenablog.com/entry/2020/12/25/123930)

### ディスカッション
いくつかディスカッションを立てて議論しました。
- [Nameのembeddings表現から特徴量を得られないか検討](https://www.guruguru.science/competitions/13/discussions/61d5d281-34ec-45a7-8cd1-70f87e5eda2c/)
- [Publisher == “Unknown” のデータに対する考察](https://www.guruguru.science/competitions/13/discussions/18ae78f3-45bb-4cea-839f-a7f7b561c464/)

---

### 案メモ
- [ ] 市場規模
- [ ] 数年後に違うPlatformで発売されたか？（人気作品はリメイクや廉価版が発売されることが多いです）
- [ ] 'Developer','Rating'に対するなんらかの処理
- [ ] いくつか発売年がおかしいデータが存在しているのでそこに対処する
- [ ] 各種エンコーディングの処理をまとめて行う