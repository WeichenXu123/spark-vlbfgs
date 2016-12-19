/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.optim

import breeze.optimize.{BacktrackingLineSearch, DiffFunction}
import org.apache.spark.ml.linalg.{Vector, Vectors, DistributedVector => DV}
import org.apache.spark.storage.StorageLevel

/**
 * Implements the Orthant-wise Limited Memory QuasiNewton method based on `VectorFreeLBFGS`,
 * which is a variant of LBFGS that handles L1 regularization.
 *
 * Paper is Andrew and Gao (2007) Scalable Training of L1-Regularized Log-Linear Models
 */
class VectorFreeOWLQN (
    maxIter: Int,
    m: Int,
    l1RegValue: Double,
    l1RegDV: DV,
    tolerance: Double,
    eagerPersist: Boolean)

  extends VectorFreeLBFGS(maxIter, m, tolerance, eagerPersist) { optimizer =>

  def this(maxIter: Int, m: Int, l1Reg: DV, tolerance: Double) = {
    this(maxIter, m, 0.0, l1Reg, tolerance, eagerPersist = true)
  }

  def this(maxIter: Int, m: Int, l1Reg: Double, tolerance: Double) = {
    this(maxIter, m, l1Reg, null, tolerance, eagerPersist = true)
  }

  def this(maxIter: Int, m: Int, l1Reg: DV) = {
    this(maxIter, m, 0.0, l1Reg, tolerance = 1e-8, eagerPersist = true)
  }

  def this(maxIter: Int, m: Int, l1Reg: Double) = {
    this(maxIter, m, l1Reg, null, tolerance = 1e-8, eagerPersist = true)
  }

  override def chooseDescentDirection(history: this.History, state: this.State): DV = {
    // The original paper requires that the descent direction be corrected to be
    // in the same directional (within the same hypercube) as the adjusted gradient for proof.
    // Although this doesn't seem to affect the outcome that much in most of cases, there are some cases
    // where the algorithm won't converge (confirmed with the author, Galen Andrew).
    val dir = history.computeDirection(state.x, state.grad, state.adjustedGradient)
    val correctedDir = dir.zipPartitions(state.adjustedGradient) {
      (partDir: Vector, partAdjustedGrad: Vector) =>
        val res = new Array[Double](partDir.size)
        partDir.foreachActive{ (index: Int, dirValue: Double) =>
          val adjustedGradValue = partAdjustedGrad(index)
          res(index) = if (dirValue * adjustedGradValue < 0) dirValue else 0.0
        }
        Vectors.dense(res)
    }
    correctedDir.persist(StorageLevel.MEMORY_AND_DISK, eager = eagerPersist)
  }

  override protected def takeStep(state: State, dir: DV, stepSize: Double): DV = {
    assert(state.adjustedGradient.isPersisted)
    // projects x to be on the same orthant as y
    // this basically requires that x'_i = x_i if sign(x_i) == sign(y_i), and 0 otherwise.
    val newX = state.x.zipPartitions(dir, state.adjustedGradient) {
      (partX: Vector, partDir: Vector, partAdjGrad: Vector) =>
        val res = new Array[Double](partX.size)
        var i = 0
        while (i < partX.size) {
          val vX = partX(i)
          val vDir = partDir(i)
          val vAdjGrad = partAdjGrad(i)

          val orthant = if (vX != 0) math.signum(vX) else math.signum(-vAdjGrad)
          val stepped = vX + vDir * stepSize

          res(i) = if (math.signum(stepped) == math.signum(orthant)) stepped else 0.0
          i += 1
        }
        Vectors.dense(res)
    }
    newX.persist(StorageLevel.MEMORY_AND_DISK, eager = true)
  }

  // Adds in the regularization stuff to the gradient(pseudo-gradient)
  override protected def adjust(newX: DV, newGrad: DV, newVal: Double): (Double, DV) = {
    /**
     * calculate objective L1 reg value contributed by this component,
     *  and the component of pseudo gradient with L1
     */
    val calculateComponentWithL1 = (vX: Double, vGrad: Double, l1Reg: Double) => {
      require(l1Reg >= 0.0)
      if (l1Reg == 0.0) (0.0, vGrad)
      else {
        val l1RegItem = Math.abs(l1Reg * vX)

        val vGradWithL1 = vX match {
          case 0.0 => {
            val delta_+ = vGrad + l1Reg
            val delta_- = vGrad - l1Reg
            if (delta_- > 0) delta_- else if (delta_+ < 0) delta_+ else 0.0
          }
          case _ => vGrad + math.signum(vX) * l1Reg
        }

        (l1RegItem, vGradWithL1)
      }
    }

    val l1ValueAccu = newX.vecs.sparkContext.doubleAccumulator
    val adjGrad = if (l1RegDV != null) {
      newX.zipPartitions(newGrad, l1RegDV) {
        (partX: Vector, partGrad: Vector, partReg: Vector) =>
          val res = Array.fill(partX.size)(0.0)
          var i = 0
          while (i < partX.size) {
            val (l1Value, vAdjGrad) = calculateComponentWithL1(partX(i), partGrad(i), partReg(i))
            l1ValueAccu.add(l1Value)
            res(i) = vAdjGrad
            i += 1
          }
          Vectors.dense(res)
      }
    } else {
      val localL1RegValue = l1RegValue
      newX.zipPartitions(newGrad) {
        (partX: Vector, partGrad: Vector) =>
          val res = Array.fill(partX.size)(0.0)
          var i = 0
          while (i < partX.size) {
            val (l1Value, vAdjGrad) =
              calculateComponentWithL1(partX(i), partGrad(i), localL1RegValue)
            l1ValueAccu.add(l1Value)
            res(i) = vAdjGrad
            i += 1
          }
          Vectors.dense(res)
      }
    }
    // Here must use eager persist because we need get the `l1ValueAccu` value immediately
    adjGrad.persist(StorageLevel.MEMORY_AND_DISK, eager = true)

    val adjValue = newVal + l1ValueAccu.value
    (adjValue, adjGrad)
  }

  // Line Search DiffFunction for vector-free OWLQN
  class VOWLQNLineSearchDiffFun(
      state: this.State,
      direction: DV,
      outer: VDiffFunction,
      eagerPersist: Boolean)
    extends DiffFunction[Double]{

    // store last point vector
    var lastX: DV = null

    // store last gradient vector
    var lastGrad: DV = null

    // store last adjusted gradient vector
    var lastAdjGrad: DV = null

    // calculates the value at a point
    override def valueAt(alpha: Double): Double = calculate(alpha)._1

    // calculates the gradient at a point
    override def gradientAt(alpha: Double): Double = calculate(alpha)._2

    // Calculates both the value and the gradient at a point
    def calculate(alpha: Double): (Double, Double) = {

      // release unused RDDs
      disposeLastResult()

      // Note: here must call OWLQN.takeStep
      lastX = optimizer.takeStep(state, direction, alpha)

      val (fnValue, grad) = outer.calculate(lastX)
      assert(grad.isPersisted)

      lastGrad = grad

      // Note: here must call OWLQN.adjust
      val (adjValue, adjGrad) = optimizer.adjust(lastX, lastGrad, fnValue)
      assert(adjGrad.isPersisted)

      lastAdjGrad = adjGrad

      val lineSearchFnGrad = adjGrad dot direction
      adjValue -> lineSearchFnGrad
    }

    def disposeLastResult() = {
      // release last point vector
      if (lastX != null) {
        lastX.unpersist()
        lastX = null
      }

      // release last gradient vector
      if (lastGrad != null) {
        lastGrad.unpersist()
        lastGrad = null
      }

      // release last adjusted gradient vector
      if (lastAdjGrad != null) {
        lastAdjGrad.unpersist()
        lastAdjGrad = null
      }
    }
  }

  override protected def determineStepSize(
      state: State,
      fn: VDiffFunction,
      direction: DV): Double = {

    val diffFun = new VOWLQNLineSearchDiffFun(state, direction, fn, eagerPersist)

    val iter = state.iter
    val search = new BacktrackingLineSearch(state.value, shrinkStep = if(iter < 1) 0.1 else 0.5)
    val alpha = search.minimize(
      diffFun,
      if (iter < 1) 0.5 / state.grad.norm() else 1.0
    )
    alpha
  }
}