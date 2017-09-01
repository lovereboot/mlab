package com.alonso.mlab.spark.basic

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

abstract class BaseSparkCase {
  var conf: SparkConf = null
  var sc: SparkContext = null
  /**
    * 资源文件路径前缀
    */
  val _RES_PREFIX: String = getClass.getResource("/").getPath


  def main(args: Array[String]): Unit = {
    initSpark()
    algorithmCase()
    sc.stop()
  }

  def initSpark(): Unit = {
    conf = new SparkConf().setMaster("local").setAppName(getAppName)
    sc = new SparkContext(conf)
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

  /**
    * 算法实例实现
    */
  def algorithmCase(): Unit

  /**
    * spark应用名称
    *
    * @return
    */
  def getAppName: String
}
