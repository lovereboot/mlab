package com.alonso.mlab.spark.basic

import breeze.linalg.sum
import org.apache.spark.rdd.RDD

object Console extends BaseSparkCase {
  /**
    * 算法实例实现
    */
  override def algorithmCase(): Unit = {
    val a1 = Array(1,2,3,4)
    val a2 = Array(4, 5,6)
    val a3 = a1++a2
    a3.foreach(println)
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
