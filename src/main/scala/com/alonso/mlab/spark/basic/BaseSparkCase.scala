package com.alonso.mlab.spark.basic

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

abstract class BaseSparkCase {
  /**
    * 资源文件路径前缀
    */
  val _RES_PREFIX: String = "mlab/target/classes/"

  val conf = new SparkConf().setMaster("local").setAppName(getAppName)
  val sc = new SparkContext(conf)

  def main(args: Array[String]): Unit = {
    algorithmCase()
    sc.stop()
  }

  /**
    * 加载资源文件
    *
    * @param path resource目录下相对路径
    * @return
    */
  def loadResourceFile(path: String): RDD[String] = {
    sc.textFile(_RES_PREFIX + path)
  }

  def algorithmCase(): Unit

  def getAppName: String
}
