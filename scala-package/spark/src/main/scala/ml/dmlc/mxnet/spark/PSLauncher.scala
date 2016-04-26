package ml.dmlc.mxnet.spark

import java.io.{PrintStream, File}

import org.apache.tools.ant.taskdefs.{Java, Echo}
import org.apache.tools.ant.types.{FileSet, Path}
import org.apache.tools.ant.{DemuxOutputStream, Project, DefaultLogger}

object PSLauncher {
  def launch(role: String, numWorker: Int = 1, spawn: Boolean = false,
    classpath: String = "/Users/lewis/Workspace/source-codes/forks/mxnet/scala-package"): Unit = {
    // global ant project settings
    val project = new Project()
    project.setBaseDir(new File(System.getProperty("user.dir")))
    project.init()
    val logger = new DefaultLogger()
    project.addBuildListener(logger)
    logger.setOutputPrintStream(System.out)
    logger.setErrorPrintStream(System.err)
    logger.setMessageOutputLevel(Project.MSG_INFO)
    System.setOut(new PrintStream(new DemuxOutputStream(project, false)))
    System.setErr(new PrintStream(new DemuxOutputStream(project, true)))
    project.fireBuildStarted()

    // an echo example
    val echo = new Echo()
    echo.setTaskName("Echo")
    echo.setProject(project)
    echo.init()
    echo.setMessage(s"Launching $role ...")
    echo.execute()

    /** initialize an java task **/
    val javaTask = new Java()
    javaTask.setNewenvironment(true)
    javaTask.setTaskName("runjava")
    javaTask.setProject(project)
    javaTask.setFork(true)
    javaTask.setFailonerror(true)

    // add some vm args
    val jvmArgs = javaTask.createJvmarg()
    jvmArgs.setLine("-Xms512m -Xmx512m")

    // added some args for to class to launch
    val taskArgs = javaTask.createArg()
    taskArgs.setLine(numWorker.toString)

    /** set the class path */
    //val classDir = new File(System.getProperty("user.dir"), "classes")
    val classDir = new File(classpath)
    val classPath = new Path(project)
    classPath.setPath(classDir.getPath)
    val fileSet = new FileSet()
    fileSet.setDir(classDir)
    fileSet.setIncludes("**/*.jar")
    classPath.addFileset(fileSet)
    javaTask.setClasspath(classPath)

    if (role == "server") {
      println("ClassLocation: "
        + classOf[PSServer].getProtectionDomain().getCodeSource().getLocation())
      javaTask.setClassname(classOf[PSServer].getName)
    } else if (role == "scheduler") {
      println("ClassLocation: "
        + classOf[PSServer].getProtectionDomain().getCodeSource().getLocation())
      javaTask.setClassname(classOf[PSScheduler].getName)
    } else {
      javaTask.setClassname(classOf[PSWorker].getName)
    }

    // this can prevent spark from stage blocking
    javaTask.setSpawn(spawn)
    javaTask.init()
    val ret = javaTask.executeJava()
    println("return code: " + ret)
    project.log("finished")
  }
}
