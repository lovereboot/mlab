package com.alonso.mlab.spark.basic

import breeze.linalg.sum
import org.apache.spark.rdd.RDD

object Console extends BaseSparkCase {
  /**
    * 算法实例实现
    */
  override def algorithmCase(): Unit = {
    val array1 = List(Array("A", "B", "C"))
    val rdd = loadResourceFile("data.txt").map(x => x.split(","))
//    rdd.foreach(x=> x.foreach(print))
    val data = for (i <- Range(0,3)) yield getMapping(rdd, i)

    getMapping(rdd, 1).foreach(println)
    data.foreach(println)
    println(sum(data.map(_.size)))
  }

  def getMapping(rdd: RDD[Array[String]], idx: Int) = {
    rdd.map(row => row(idx)).distinct().zipWithIndex().collectAsMap()
  }

  /**
    * spark应用名称
    *
    * @return
    */
  override def getAppName: String = "ConsoleApp"
}
