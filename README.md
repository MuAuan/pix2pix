# pix2pix
有名なpix2pixの検証：GANの一種

Keras で pix2pix を実装する。【 cGAN 考慮】ｂｙ　トミーさん

http://toxweblog.toxbe.com/2017/12/24/keras-%e3%81%a7-pix2pix-%e3%82%92%e5%ae%9f%e8%a3%85/

の検証してみました。

img2h5.pyが画像の入ったフォルダからh5pyファイルを生成するプログラムです
　入力画像の大きさに依存しないように書き換えました。

使い方は以下のサイトを参照してください。

tommyfms2/pix2pix-keras-byt 

https://github.com/tommyfms2/pix2pix-keras-byt

簡単に説明すると以下の通りです。

### １．img2h5.pyで入力ファイルを作成する

　実行は以下のとおり、images/にカテゴリ毎にディレクトリを作成して画像を配置しています。
 
　python img2h5.py -i images/ -o datasetimages -t canny
 
 出力は、datasetimgesというファイルが作成されます。
 
　※画像はカテゴリ毎に読み込みと合わせて配置してください。　
 
　※画像サイズはバラバラでも対応しています、サイズ変更されてしまうので、
 
　　　元画像を保存したい場合はあらかじめコピーするか、別途ディレクトリを決めてそこに保存・取得するように変更してください。

### ２．Trainingは以下で実行

　python pix2pix.py -d datasetimages.hdf5

オプションがあるみたいですが、使いませんでした。

 上記サイトを参照願います
 
 ### ３．出力はTrainigとVaridationの二つがでます。
 
 　／figureに格納しました。
  
 　４００回回して、比較的きれいな以下のようなものが出力されます。

   どちらも、
   
  　上段が元画像をCannyした線画
    中断がGenerateされた画像
    下段が元画像
   
   です。
     
# pix2pix_alt.py

### ①データをTrain、Test分離して入力

### ②parameterの入出力対応

### ③結果を間引いて出力

## データ作成をTrain、Testに分けて作成

python img2h5_4TrainData.py -i images/ -o dataset4Train -t canny 

python img2h5_4TestData.py -i images/ -o dataset4Test -t canny 

## models.pyにencoder-decoderも実行できるように改変
def up_conv_block_unet_alt(x, x2, f, name, bn_axis, bn=True, dropout=False):
を追加した。
