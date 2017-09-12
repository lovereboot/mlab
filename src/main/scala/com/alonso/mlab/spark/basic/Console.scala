package com.alonso.mlab.spark.basic

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StringIndexer

object Console extends BaseSparkCase {
  /**
    * 算法实例实现
    */
  override def algorithmCase(): Unit = {
    val df = loadCsvDf("data.txt").toDF("c1", "c2", "c3")
    df.show()
    val strIdx01 = new StringIndexer().setInputCol("c1").setOutputCol("c1_idx")
    val strIdx02 = new StringIndexer().setInputCol("c2").setOutputCol("c2_idx")

    val p1 = new Pipeline().setStages(Array(strIdx01))

    val p2 = new Pipeline().setStages(Array(strIdx01, strIdx02))
    val pRes1 = p1.fit(df)
    val pRes2 = p2.fit(df)
    val r1 = pRes1.transform(df)
    val r2 = pRes2.transform(df)

    r1.show()
    r2.show()
  }

  /**
    * spark应用名称
    *
    * @return
    */
  override def getAppName: String = "ConsoleApp"
}
