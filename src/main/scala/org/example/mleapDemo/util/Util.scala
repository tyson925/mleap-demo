package org.example.mleapDemo.util

import java.io.{BufferedWriter, File, FileWriter}

object Util {


  def writeContent(content : String, fileName : String): Unit ={
    val file = new File(fileName)
    val bw = new BufferedWriter(new FileWriter(file))
    //bw.write(content.mkString("\n"))
    bw.write(content.mkString)
    bw.close()
  }
}
