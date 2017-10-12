package org.example.mleapDemo.customTransformer.mleap

import ml.combust.mleap.core.Model
import ml.combust.mleap.core.types.{BasicType, ListType, StructType}


case class WordLengthFilterModel(length: Int = 3) extends Model {
  def apply(words: Seq[String]): Seq[String] =  {
    words.filter(word =>
      word.forall(w =>
        w.isLetter) && word.length > length)
  }

  override def inputSchema: StructType = StructType("input" -> ListType(BasicType.String)).get

  override def outputSchema: StructType = StructType("output" -> ListType(BasicType.String)).get
}
