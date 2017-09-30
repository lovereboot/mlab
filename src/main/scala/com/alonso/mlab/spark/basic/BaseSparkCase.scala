package com.alonso.mlab.spark.basic

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

abstract class BaseSparkCase {
  var conf: SparkConf = null
  var sc: SparkContext = null
  var session: SparkSession = null

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
    session = SparkSession.builder().master("local").appName(getAppName).config("spark.some.config.option", "some-value").getOrCreate()
  }

  /**
    * 加载资源文件
    *
    * @param path resource目录下相对路径
    * @return
    */
  def loadRdd(path: String): RDD[String] = {
    sc.textFile(_RES_PREFIX + path)
  }

  /**
    * 加载数据
    *
    * @param path
    * @return
    */
  def loadCsvDf(path: String): DataFrame = {
    loadCsvDf(_RES_PREFIX + path, false)
  }

  /**
    * 加载数据
    *
    * @param path
    * @param hasTitle 数据是否含标题
    * @return
    */
  def loadCsvDf(path: String, hasTitle: Boolean): DataFrame = {
    val reader = session.read
    if (hasTitle) {
      reader.option("header", true)
    }
    reader.csv(path)
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
