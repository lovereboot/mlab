package com.alonso.mlab.spark.basic

import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors

/**
  * Kmeans算法：<br/>
  * <li>1. 随机选择k个点作为初始种子点，也就是需要分的组数</li>
  * <li>2. 计算每个点到种子点的距离，如果p点距离种子点s1最近，那么p点属于s1点群</li>
  * <li>3. 将种子点移动到属于他点群点群的中心</li>
  * <li>4. 重复2、3步，直到种子点不再移动</li>
  */
object KmeansApp extends BaseSparkCase {

  override def getAppName = "KmeansApp"

  override def algorithmCase(): Unit = {
    // 装载数据
    val file = loadResourceFile("kmeans/kmeans_data.txt")
    val parsedData = file.map(s => Vectors.dense(s.split(" ").map(_.toDouble)))

    val numClusters = 2
    val numIterations = 20
    val model = KMeans.train(parsedData, numClusters, numIterations) // 模型训练，分2组，最多20次迭代

    // 打印数据模型的中心点
    println("Cluster centers:")
    for (c <- model.clusterCenters) {
      println(" " + c.toString)
    }

    // 使用方差来评估模型
    val cost = model.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + cost)

    // 使用模型测试单点数据
    println("Vectors 0.2 0.2 0.2 is belongs to clusters:" + model.predict(Vectors.dense("0.2 0.2 0.2".split(" ").map(_.toDouble))))
    println("Vectors 0.25 0.25 0.25 is belongs to clusters:" + model.predict(Vectors.dense("0.25 0.25 0.25".split(' ').map(_.toDouble))))
    println("Vectors 8 8 8 is belongs to clusters:" + model.predict(Vectors.dense("8 8 8".split(' ').map(_.toDouble))))

    // 交叉评估
    val testData = file.map(line => Vectors.dense(line.split(" ").map(_.toDouble)))
    model.predict(testData).foreach(println)

    file.map {
      line =>
        val linevectors = Vectors.dense(line.split(" ").map(_.toDouble))
        val predication = model.predict(linevectors)
        line + " " + predication
    }.foreach(println)
  }
}
