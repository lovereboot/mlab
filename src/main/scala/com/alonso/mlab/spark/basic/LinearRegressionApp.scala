package com.alonso.mlab.spark.basic

import breeze.linalg.sum
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.rdd.RDD

/**
  * 线性回归算法
  */
object LinearRegressionApp extends BaseSparkCase {
  /**
    * 算法实例实现
    */
  override def algorithmCase(): Unit = {
    linearRegressionCase_01
  }

  /**
    * 线性回归算法场景一：自行车租车量预测<br/>
    * 训练数据line_regression_hour.csv列标题：记录Id,日期,季节,年份,月份,时间,是否节假日,星期,是否工作日,天气类型,气温,体感温度,风速,每小时租车量
    */
  def linearRegressionCase_01(): Unit = {
    // 1.加载数据
    val file = loadResourceFile("regression/linear/line_regression_hour.csv")
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

      for (typeVal <- row.slice(2, 10)) {
        val enums = typeEnums(i) // 取出第i行类型变量枚举列表,数据类型<code>Map[String, Long]</code>

        val idx = enums(typeVal) // 取出类型变量枚举值zipWithIndex
        catVec(idx.toInt + step) = 1.0 // 类别特征数据处理，根据枚举类别转换为特征值
        i = i + 1
        step = step + enums.size
      }

      catVec.foreach(println)

      val numVec = row.slice(10, 14).map(x => x.toDouble)
      val features = catVec ++ numVec
      val label = row(row.size - 1).toInt // 自行车租车量
      LabeledPoint(label, Vectors.dense(features))
    }

    val linearModel = LinearRegressionWithSGD.train(data, 10, 1.0)

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
    rdd.map(row => row(idx)).distinct().zipWithIndex().collectAsMap()
  }

  /**
    * spark应用名称
    *
    * @return
    */
  override def getAppName: String = "LinearRegressionApp"
}
