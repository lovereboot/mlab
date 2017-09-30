package com.alonso.mlab.spark.basic

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

/**
  * 随机森林算法
  */
object RandomForestApp extends BaseSparkCase {
  /**
    * 算法实例实现
    */
  override def algorithmCase(): Unit = {
    val data = session.read.option("header", true).csv(_RES_PREFIX + "rdmf/random-forest.txt")
    val creditDf = session.createDataFrame(parseRDD(data).map(parseCredit))
    val colNames = creditDf.schema.toArray.map(_.name)
    val featureCols = colNames.slice(1, colNames.length)
    // 训练模型数据特征值提取
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val df2 = assembler.transform(creditDf)
    // 训练结果数据标准化(对于该实例数据，该步骤主要针对结果为字符串的训练数据，当前训练数据结果已经标准化,可以去掉该步骤)
    val labelIndexer = new StringIndexer().setInputCol("creditability").setOutputCol("label")
    val df3 = labelIndexer.fit(df2).transform(df2) // 数据标准化。creditability出现频次，转换成0～num numOfLabels-1(分类个数)，频次最高的转换为0

    val Array(trainData, predictData) = df3.randomSplit(Array(0.7, 0.3), seed = 5043L)

    val classifier = new RandomForestClassifier().setImpurity("gini").setMaxDepth(3).setNumTrees(5).setFeatureSubsetStrategy("auto").setSeed(55043)
    val model = classifier.fit(trainData)

    val predictResult = model.transform(predictData)

    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction")
    val accuracy = evaluator.evaluate(predictResult) // 评估训练模型预测准确度
    println("current model accuracy : %s".format(accuracy))

    // 构建pipeline,优化模型训练
    val paramGridBuilder = new ParamGridBuilder()
      .addGrid(classifier.maxBins, Array(25, 31)) // 连续特征离散化的最大数量，以及选择每个节点分裂特征的方式
      .addGrid(classifier.numTrees, Array(20, 60))
      .addGrid(classifier.maxDepth, Array(5, 10))
      .addGrid(classifier.impurity, Array("entropy", "gini")).build()

    val pipeline = new Pipeline().setStages(Array(classifier))

    val cv = new CrossValidator() // 交叉验证
      .setEvaluator(evaluator)
      .setEstimator(pipeline)
      .setEstimatorParamMaps(paramGridBuilder)
      .setNumFolds(10) // 10折交叉验证

    val pipelineFittedModel = cv.fit(trainData)

    val cvPredictResult = pipelineFittedModel.transform(predictData)
    val f1 = evaluator.evaluate(cvPredictResult)
    println("cross validator model accuracy : %s".format(f1))
  }

//  def parseCredit(line: Array[Double]): Credit = {
//    Credit(
//      line(0),
//      line(1) - 1, line(2), line(3), line(4), line(5),
//      line(6) - 1, line(7) - 1, line(8), line(9) - 1, line(10) - 1,
//      line(11) - 1, line(12) - 1, line(13), line(14) - 1, line(15) - 1,
//      line(16) - 1, line(17) - 1, line(18) - 1, line(19) - 1, line(20) - 1
//    )
//  }

  def parseCredit(line: Array[Double]): Credit = {
    Credit(
      line(0),
      line(1), line(2), line(3), line(4), line(5),
      line(6), line(7), line(8), line(9), line(10),
      line(11), line(12), line(13), line(14), line(15),
      line(16), line(17), line(18), line(19), line(20)
    )
  }

  def parseRDD(rdd: DataFrame): RDD[Array[Double]] = {
    rdd.rdd.map { row =>
      val array = new Array[Double](row.length)
      for (i <- Range(0, row.length)) array(i) = row.get(i).toString.toDouble
      array
    }
  }

  case class Credit(creditability: Double,
                    balance: Double, // 活期存款
                    duration: Double, // 贷款期限
                    history: Double, // 贷款历史(无贷款/均按时还款;当前银行的所有贷款都按时还款;存在贷款且到目前为止都按时还款;有逾期还款经历;危险账户/存在他行贷款)
                    purpose: Double, // 贷款性质
                    amount: Double, // 贷款数额
                    savings: Double, // 储蓄账户存款
                    employment: Double, // 工作年限
                    instPercent: Double, // 可支配收入百分比
                    sexMarried: Double, // 性别&婚姻状况
                    guarantors: Double, // 保证人
                    residenceDuration: Double, // 当前地址居住年限
                    assets: Double, // 个人资产
                    age: Double, // 年龄
                    concCredit: Double,
                    apartment: Double, // 住宅类型
                    credits: Double, // 当前银行贷款情况
                    occupation: Double,
                    dependents: Double,
                    hasPhone: Double,
                    foreign: Double
                   )

  /**
    * spark应用名称
    *
    * @return
    */
  override def getAppName: String = "RandomForestApp"
}
