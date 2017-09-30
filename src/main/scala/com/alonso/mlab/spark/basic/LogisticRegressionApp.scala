package com.alonso.mlab.spark.basic

import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint


object LogisticRegressionApp extends BaseSparkCase {
  /**
    * 逻辑回归case：预测企业是否会破产。输入数据列格式如下：
    *
    * <li>工业风险：取值P|A|N</li>
    * <li>管理风险：取值P|A|N</li>
    * <li>财务灵活性：取值P|A|N</li>
    * <li>信誉：取值P|A|N</li>
    * <li>竞争力：取值P|A|N</li>
    * <li>经营风险：取值P|A|N</li>
    *
    * <li>公司定性分类：取值B：破产;NB：非破产</li>
    *
    */
  override def algorithmCase(): Unit = {
    val file = loadRdd("regression/logistic/logistic.txt")
    val data = file.map { row =>
      val cols = row.split(",")
      LabeledPoint(toDouble(cols(6)), Vectors.dense(cols.slice(0, 6).map(col => toDouble(col))))
    }
    val split = data.randomSplit(Array(0.7, 0.3), seed = 11L)
    val train = split(0)
    val predict = split(1)

    val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(train)
    val predictResult = predict.map { point =>
      val result = model.predict(point.features)
      (point.label, result)
    }
    session.createDataFrame(predictResult).toDF("real_result", "predict_result").show(50)
  }

  def rfModel(): Unit = {
    val file = loadCsvDf("regression/logistic/logistic.txt")
    val data = file.toDF("industry_risk", "")
  }

  def toDouble(input: String): Double = {
    var result: Double = 0.0
    if ("P".equalsIgnoreCase(input)) {
      result = 3.0
    } else if ("A".equalsIgnoreCase(input)) {
      result = 2.0
    } else if ("N".equalsIgnoreCase(input)) {
      result = 1.0
    } else if ("NB".equalsIgnoreCase(input)) {
      result = 1.0
    } else if ("B".equalsIgnoreCase(input)) {
      result = 0.0
    }
    result
  }

  /**
    * spark应用名称
    *
    * @return
    */
  override def getAppName: String = "LogisticRegressionApp"
}
