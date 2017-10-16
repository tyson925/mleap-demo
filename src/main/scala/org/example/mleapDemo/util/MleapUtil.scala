package org.example.mleapDemo.util

import java.io.File

import ml.combust.bundle.BundleFile
import ml.combust.bundle.serializer.SerializationFormat
import ml.combust.mleap.spark.SparkSupport._
import org.apache.spark.ml
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.bundle.SparkBundleContext
import org.apache.spark.sql.DataFrame
import resource.managed

object MleapUtil {

  def writePipeline(pipelineModel: PipelineModel, dataFrame: DataFrame, fileName: String): Unit = {
    val file = new File(fileName)

    if (file.exists()) {
      println("Update model...")
      file.delete()
    }
    file.getParentFile.mkdirs
    //IT is imprtant, it is required to storing the model
    val sbc = SparkBundleContext().withDataset(pipelineModel.transform(dataFrame))
    for (bundle <- managed(BundleFile(file))) {
      pipelineModel.writeBundle.format(SerializationFormat.Json).save(bundle)(sbc)
      //parkPipelineLr.writeBundle.save(bf)
    }
  }

  def deserializeModel(fileName: String): ml.Transformer = {
    //val zipBundle = (for (bundle <- managed(BundleFile(new URI("./data/mleap-examples/simple-json.zip")))) yield {
    //  bundle.loadSparkBundle().get
    //}).opt.get

    val zipBundle = (for (bundle <- managed(BundleFile(new File(fileName)))) yield {
      bundle.loadSparkBundle().get
    })

    zipBundle.opt.get.root
  }
}
