package org.apache.spark.example.ml

import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{DistributedVectorPartitioner, Vector, Vectors, DistributedVector => DV, DistributedVectors => DVs}
import org.apache.spark.ml.optim.{DVDiffFunction, VectorFreeLBFGS}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel


class SimpleVFLogisticRegressionDiffFunction (
                                            sc: SparkContext,
                                            labeledData: RDD[LabeledPoint],
                                            _numInstances: Long,
                                            _featureSize: Int
                                            ) extends DVDiffFunction {
  // Calculates both the value and the gradient at a point
  override def calculate(x: DV): (Double, DV) = {
    val numInstances = _numInstances
    val featureSize = _featureSize
    val lossSumAccu = sc.doubleAccumulator
    val gradRdd = labeledData.cartesian(x.vecs).map{
      case (labeledPoint: LabeledPoint, coeffs: Vector) => {
        var margin = 0.0
        labeledPoint.features.foreachActive((idx: Int, value: Double) => {
          margin += (value * coeffs(idx))
        })
        margin = -margin
        val multiplier = 1.0 / (1.0 + math.exp(margin)) - labeledPoint.label
        lossSumAccu.add(
          if (labeledPoint.label > 0) { MLUtils.log1pExp(margin) }
          else { MLUtils.log1pExp(margin) - margin }
        )
        (0, labeledPoint.features.asBreeze.toDenseVector * multiplier)
      }
    }.reduceByKey(new DistributedVectorPartitioner(1), _ + _)
      .map(kv => Vectors.fromBreeze(kv._2 / (numInstances * 1.0)))
      .persist(StorageLevel.MEMORY_AND_DISK)
    gradRdd.count() // force persist.
    val loss = lossSumAccu.value / numInstances
    println(s"vf iter: loss: ${loss}, grad: ${gradRdd.collect()(0)}")
    (loss, new DV(gradRdd, featureSize, 1, featureSize).eagerPersist())
  }
}

class SimpleVFLogisticRegression(val _maxIter: Int = 100) {
  def train(dataset: Dataset[_]): DV = {
    val maxIter = _maxIter
    val sc = dataset.sparkSession.sparkContext
    val labeledData: RDD[LabeledPoint] =
      dataset.select(col("label"), col("features")).rdd.map {
        case Row(label: Double, features: Vector) =>
          LabeledPoint(label, features)
      }.persist(StorageLevel.MEMORY_AND_DISK)
    val numInstances = labeledData.count()
    val featureSize = labeledData.first().features.size
    println(s"vf: numInstances: ${numInstances}, featureSize: ${featureSize}")
    val costFun = new SimpleVFLogisticRegressionDiffFunction(sc, labeledData, numInstances, featureSize)
    val initCoeffs = DVs.zeros(sc, featureSize, 1, featureSize).eagerPersist()
    val optimizer = new VectorFreeLBFGS(maxIter, 10)
    val states = optimizer.iterations(costFun, initCoeffs)
    var state: optimizer.State = null
    while (states.hasNext) {
      state = states.next()
    }
    state.x // `x` already persisted.
  }
}

object SimpleVFLogisticRegression {
  def main(args: Array[String]) = {
    val spark = SparkSession.builder().appName("SimpleVFLogisticRegressionExample").getOrCreate()
    val sc = spark.sparkContext
    val data1 = Seq(
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, -2.1, -1.5).toSparse),
      LabeledPoint(0.0, Vectors.dense(0.9, 3.0, -2.1, -1.1).toSparse),
      LabeledPoint(0.0, Vectors.dense(1.0, 2.0, 0.0, 1.2).toSparse),
      LabeledPoint(1.0, Vectors.dense(-1.5, 1.0, -0.5, 0.0).toSparse),
      LabeledPoint(1.0, Vectors.dense(-1.9, 1.5, -0.3, -1.5).toSparse),
      LabeledPoint(1.0, Vectors.dense(1.1, 1.0, -2.1, -1.5).toSparse),
      LabeledPoint(0.0, Vectors.dense(0.8, 3.0, -2.1, -1.1).toSparse),
      LabeledPoint(0.0, Vectors.dense(1.2, 2.0, 0.0, 1.2).toSparse),
      LabeledPoint(1.0, Vectors.dense(-1.3, 1.0, -0.5, 0.0).toSparse),
      LabeledPoint(1.0, Vectors.dense(-1.5, 1.5, -0.3, -1.5).toSparse),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, -2.1, -1.5).toSparse),
      LabeledPoint(0.0, Vectors.dense(0.9, 3.0, -2.1, -1.1).toSparse),
      LabeledPoint(0.0, Vectors.dense(1.0, 2.0, 0.0, 1.2).toSparse),
      LabeledPoint(1.0, Vectors.dense(-1.5, 1.0, -0.5, 0.0).toSparse),
      LabeledPoint(1.0, Vectors.dense(-1.9, 1.5, -0.3, -1.5).toSparse),
      LabeledPoint(1.0, Vectors.dense(1.1, 1.0, -2.1, -1.5).toSparse),
      LabeledPoint(0.0, Vectors.dense(0.8, 3.0, -2.1, -1.1).toSparse),
      LabeledPoint(0.0, Vectors.dense(1.2, 2.0, 0.0, 1.2).toSparse),
      LabeledPoint(1.0, Vectors.dense(-1.3, 1.0, -0.5, 0.0).toSparse),
      LabeledPoint(1.0, Vectors.dense(-1.5, 1.5, -0.3, -1.5).toSparse),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, -2.1, -1.5).toSparse),
      LabeledPoint(0.0, Vectors.dense(0.9, 3.0, -2.1, -1.1).toSparse),
      LabeledPoint(0.0, Vectors.dense(1.0, 2.0, 0.0, 1.2).toSparse),
      LabeledPoint(1.0, Vectors.dense(-1.5, 1.0, -0.5, 0.0).toSparse),
      LabeledPoint(1.0, Vectors.dense(-1.9, 1.5, -0.3, -1.5).toSparse),
      LabeledPoint(1.0, Vectors.dense(1.1, 1.0, -2.1, -1.5).toSparse),
      LabeledPoint(0.0, Vectors.dense(0.8, 3.0, -2.1, -1.1).toSparse),
      LabeledPoint(0.0, Vectors.dense(1.2, 2.0, 0.0, 1.2).toSparse),
      LabeledPoint(1.0, Vectors.dense(-1.3, 1.0, -0.5, 0.0).toSparse),
      LabeledPoint(1.0, Vectors.dense(-1.5, 1.5, -0.3, -1.5).toSparse),
      LabeledPoint(1.0, Vectors.dense(1.0, 1.0, -2.1, -1.5).toSparse),
      LabeledPoint(0.0, Vectors.dense(0.9, 3.0, -2.1, -1.1).toSparse),
      LabeledPoint(0.0, Vectors.dense(1.0, 2.0, 0.0, 1.2).toSparse),
      LabeledPoint(1.0, Vectors.dense(-1.5, 1.0, -0.5, 0.0).toSparse),
      LabeledPoint(1.0, Vectors.dense(-1.9, 1.5, -0.3, -1.5).toSparse),
      LabeledPoint(1.0, Vectors.dense(1.1, 1.0, -2.1, -1.5).toSparse),
      LabeledPoint(0.0, Vectors.dense(0.8, 3.0, -2.1, -1.1).toSparse),
      LabeledPoint(0.0, Vectors.dense(1.2, 2.0, 0.0, 1.2).toSparse),
      LabeledPoint(1.0, Vectors.dense(-1.3, 1.0, -0.5, 0.0).toSparse),
      LabeledPoint(1.0, Vectors.dense(-1.5, 1.5, -0.3, -1.5).toSparse)
    )
    val dataset1 = spark.createDataFrame(sc.parallelize(data1, 40).mapPartitions{
      iter =>
        Thread.sleep(1000)
        iter
    })
    val vftrainer1 = new SimpleVFLogisticRegression
    val vfmodel1 = vftrainer1.train(dataset1)
    val breezetrainer1 = (new LogisticRegression).setFitIntercept(false)
      .setRegParam(0.0).setStandardization(false)
    val breezemodel1 = breezetrainer1.fit(dataset1)

    println(s": vf coeffs: ${vfmodel1.toLocal}\nbreeze coeffs: ${breezemodel1.coefficients}")
    Thread.sleep(1000 * 3600 * 10)
  }
}
