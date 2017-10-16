package org.example.mleapDemo.textClassification

import java.io.{BufferedWriter, File, FileWriter, FilenameFilter}

import com.fasterxml.jackson.databind.{ObjectMapper, SerializationConfig}
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import ml.combust.mleap.core.classification.DecisionTreeClassifierModel
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.mleap.SparkUtil
import org.apache.spark.ml.mleap.feature.WordLengthFilter
import org.apache.spark.sql.{DataFrame, Encoders, SparkSession}
import org.example.mleapDemo.data.{Document, ReutersParser}
import org.example.mleapDemo.util.{MleapUtil, Util}

object TextClassification {

  val mapper = new ObjectMapper().registerModule(DefaultScalaModule)

  def convertReutersDataToJson(directory: String): Unit = {
    val inputFiles = new File(directory).list(new FilenameFilter {
      override def accept(dir: File, name: String) = name.endsWith(".sgm")
    })

    val fullFileNames = inputFiles.map(directory + "/" + _)
    val corpus = ReutersParser.parseAll(fullFileNames)
    //println(docs.take(10).mkString)

    val corpusJson = corpus.map { document =>
      println(document)
      mapper.writeValueAsString(document)
    }
    Util.writeContent(corpusJson.mkString("\n"), "./data/reuters21578.json")

  }

  def createPipeline(dataFrame: DataFrame): PipelineModel = {
    val predictionCol = "label"

    val stringIndexer = {
      new StringIndexer()
        .setInputCol(predictionCol)
        .setOutputCol("label_index")
        .fit(dataFrame)
    }


    val tokenizer = new Tokenizer().setInputCol("body").setOutputCol("words")

    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("words_filtered")

    val filterWords = new WordLengthFilter().setInputCol(remover.getOutputCol).setOutputCol("filteredWords").setWordLength(3)

    val cv = new CountVectorizer().setInputCol(remover.getOutputCol).setOutputCol("features").setVocabSize(100)

    val pca = new PCA().
      setInputCol(cv.getOutputCol).
      setOutputCol("features").
      setK(100)

    val dt = new DecisionTreeClassifier().
      setFeaturesCol(cv.getOutputCol).
      setLabelCol(stringIndexer.getOutputCol).
      setPredictionCol("prediction").
      setProbabilityCol("probability").
      setRawPredictionCol("raw_prediction")

    val converter = new IndexToString()
      .setInputCol(dt.getPredictionCol)
      .setOutputCol("originalCategory")

    val ovr = new OneVsRest().setClassifier(dt).
      setFeaturesCol(cv.getOutputCol).
      setLabelCol(stringIndexer.getOutputCol).
      setPredictionCol("prediction")

    // train the multiclass model.
    //val ovrModel = ovr.fit(dataFrame)

    new Pipeline().setStages(Array(stringIndexer, tokenizer, remover, filterWords, cv, dt, converter)).fit(dataFrame)
  }

  def createRandomForest(dataFrame: DataFrame,featureColName : String, labelColName : String): RandomForestClassificationModel = {
    val rf = new RandomForestClassifier().
      setFeaturesCol(featureColName).
      setLabelCol(labelColName).
      setPredictionCol("prediction").
      setProbabilityCol("probability").
      setRawPredictionCol("raw_prediction")

    val rfModel = rf.fit(dataFrame)
    rfModel
  }

  def createDecisionTree(dataFrame: DataFrame,featureColName : String, labelColName : String): DecisionTreeClassificationModel = {
    val dt = new DecisionTreeClassifier().
      setFeaturesCol(featureColName).
    setLabelCol(labelColName).
      setPredictionCol("prediction").
      setProbabilityCol("probability").
      setRawPredictionCol("raw_prediction")

    val dtModel = dt.fit(dataFrame)
    dtModel
  }


  def main(args: Array[String]): Unit = {


    //convertReutersDataToJson("./data/reuters21578")


    val spark = SparkSession.builder().appName("lr-demo").master("local[6]").getOrCreate()

    import spark.implicits._

    /*type DocumentEncoded = (String, String, String)

    // implicit conversions
    implicit def toEncoded(document: Document): DocumentEncoded = (document.docId, document.body, document.labels.mkString)

    implicit def fromEncoded(e: DocumentEncoded): Document =
      new Document(e._1, e._2, e._3.split(" ").toSet)

    val corpusDF = spark.createDataFrame(corpusRDD, Document.getClass).toDF()*/

    val data = spark.read.json("./data/train.json")

    val featureModel = createPipeline(data)

    //val rfModel = createRandomForest(data,"pcaFeatures","label_index")

    val pipeline = SparkUtil.createPipelineModel(uid = "pipeline", Array(featureModel))

    val modelName = "./data/mleap/reuters_example.json.zip"

    MleapUtil.writePipeline(pipeline,data,modelName)

    val readModel = MleapUtil.deserializeModel(modelName)
    //corpusDF.show(1000)

    val test = spark.read.json("./data/test.json")

    val predictions = readModel.transform(test)

    predictions.select("label","originalCategory").show(10,false)

    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("f1").setLabelCol("label_index")

    // compute the classification error on test data.
    val accuracy = evaluator.evaluate(predictions)

    println(s"f1-score: ${accuracy}")
    //println(s"Test Error = ${1 - accuracy}")
  }


}
