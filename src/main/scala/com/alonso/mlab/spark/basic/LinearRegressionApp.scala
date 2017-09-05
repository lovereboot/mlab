package com.alonso.mlab.spark.basic

import breeze.linalg.sum
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

/**
  * 线性回归算法
  */
object LinearRegressionApp extends BaseSparkCase {

  val _TRAIN_DATA = "regression/linear/line_regression_hour.csv";

  /**
    * 算法实例实现
    */
  override def algorithmCase(): Unit = {
    //    spark_under_2_x_case
    spark_2_x_case
  }


  /**
    * 线性回归算法场景一：自行车租车量预测<br/>
    * 训练数据line_regression_hour.csv列标题：记录Id,日期,季节,年份,月份,时间,是否节假日,星期,是否工作日,天气类型,气温,体感温度,风速,每小时租车量
    */
  def spark_under_2_x_case(): Unit = {
    // 1.加载数据
    val file = loadResourceFile(_TRAIN_DATA)
    val records = file.map(_.split(",")).cache()
    val typeEnums = for (i <- Range(2, 10)) yield getTypeVarEnums(records, i)

    val catLen = sum(typeEnums.map(_.size)) //  8个类型变量所有枚举类型
    val numLen = records.first().slice(10, 14).size // 实数变量的个数

    println("catLen = %s; numLen = %s".format(catLen, numLen))

    val totalLen = catLen + numLen

    val data = records.map { row =>
      val catVec = Array.ofDim[Double](catLen)
      var i = 0
      var step = 0

      // 类型变量特征值处理，将类型变量转换为二元向量
      for (typeVal <- row.slice(2, 10)) {
        val enums = typeEnums(i) // 取出第i行类型变量枚举列表,数据类型<code>Map[String, Long]</code>

        val idx = enums(typeVal) // 取出类型变量枚举值zipWithIndex
        catVec(idx.toInt + step) = 1.0 // 类别特征数据处理，根据枚举类别转换为特征值
        i = i + 1
        step = step + enums.size
      }

      val numVec = row.slice(10, 14).map(x => x.toDouble)
      val features = catVec ++ numVec
      val label = row(row.size - 1).toInt // 自行车真实租车量
      LabeledPoint(label, Vectors.dense(features))
    }

    val linearModel = LinearRegressionWithSGD.train(data, 20, 0.5) // 参数:样本,迭代次数,步长

    val trueVsPredicted = data.map(p => (p.label, linearModel.predict(p.features)))

    println(trueVsPredicted.take(5).toVector.toString)

  }


  /**
    * 获取类型变量的枚举类型<br/>
    * 先将数据进行行列转换，然后去重
    *
    * @param rdd 列分布类型变量
    * @param idx 类型变量索引
    * @return
    */
  def getTypeVarEnums(rdd: RDD[Array[String]], idx: Int) = {
    rdd.map(row => row(idx)).distinct().zipWithIndex().collectAsMap() // zipWithIndex不用太care具体的index值，同行各个col唯一即可
  }

  /**
    * 在spark2.x版本，线性回归使用LinearRegression
    */
  def spark_2_x_case(): Unit = {
    val train = loadResourceFile(_TRAIN_DATA).map { row =>
      val cols = row.split(",")
      (cols(2).toDouble, cols(3).toDouble, cols(4).toDouble, cols(5).toDouble, cols(6).toDouble, cols(7).toDouble, cols(8).toDouble, cols(9).toDouble, cols(10).toDouble, cols(11).toDouble, cols(12).toDouble, cols(13).toDouble, cols(16).toDouble)
    }.cache()
    // 将数据转换为DataFrame
    val df = session.createDataFrame(train)
    // 设置DataFrame数据源列标题
    val data = df.toDF("season","yr","mnth","hr","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed","cnt")
    val inputCols = Array("season","yr","mnth","hr","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed")

    // 根据inputCols指定列提取特征值放到features中
    val assembler = new VectorAssembler().setInputCols(inputCols).setOutputCol("features")
    val inputVec = assembler.transform(data)

    val lr1 = new LinearRegression()
    val lr2 = lr1.setFeaturesCol("features").setLabelCol("cnt").setFitIntercept(true) // 是否拟合截距，默认true

    val lr3 = lr2.setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

    val lr = lr3
    val model = lr.fit(inputVec)


    // 准备预测数据
    val predictData = train.randomSplit(Array(0.7, 0.3), seed = 11L)(1)
    val predictDf = session.createDataFrame(predictData)
    val predictInput = buildData(predictDf);

    val predict = model.transform(predictInput)
    println("输出预测结果:")
    predict.selectExpr("cnt", "prediction", "features").show()
  }

  def buildData(df: DataFrame): DataFrame = {
    val data = df.toDF("season","yr","mnth","hr","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed","cnt")
    val inputCols = Array("season","yr","mnth","hr","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed")

    // 根据inputCols指定列提取特征值放到features中
    val assembler = new VectorAssembler().setInputCols(inputCols).setOutputCol("features")
    assembler.transform(data)
  }

  /**
    * spark应用名称
    *
    * @return
    */
  override def getAppName: String = "LinearRegressionApp"
}
