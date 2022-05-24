package pisa.server

object ServerGrpc {
  val METHOD_INITIALISE_ISABELLE: _root_.io.grpc.MethodDescriptor[pisa.server.IsaPath, pisa.server.IsaMessage] =
    _root_.io.grpc.MethodDescriptor.newBuilder()
      .setType(_root_.io.grpc.MethodDescriptor.MethodType.UNARY)
      .setFullMethodName(_root_.io.grpc.MethodDescriptor.generateFullMethodName("server.Server", "InitialiseIsabelle"))
      .setSampledToLocalTracing(true)
      .setRequestMarshaller(_root_.scalapb.grpc.Marshaller.forMessage[pisa.server.IsaPath])
      .setResponseMarshaller(_root_.scalapb.grpc.Marshaller.forMessage[pisa.server.IsaMessage])
      .setSchemaDescriptor(_root_.scalapb.grpc.ConcreteProtoMethodDescriptorSupplier.fromMethodDescriptor(pisa.server.ServerProto.javaDescriptor.getServices.get(0).getMethods.get(0)))
      .build()
  
  val METHOD_ISABELLE_CONTEXT: _root_.io.grpc.MethodDescriptor[pisa.server.IsaContext, pisa.server.IsaMessage] =
    _root_.io.grpc.MethodDescriptor.newBuilder()
      .setType(_root_.io.grpc.MethodDescriptor.MethodType.UNARY)
      .setFullMethodName(_root_.io.grpc.MethodDescriptor.generateFullMethodName("server.Server", "IsabelleContext"))
      .setSampledToLocalTracing(true)
      .setRequestMarshaller(_root_.scalapb.grpc.Marshaller.forMessage[pisa.server.IsaContext])
      .setResponseMarshaller(_root_.scalapb.grpc.Marshaller.forMessage[pisa.server.IsaMessage])
      .setSchemaDescriptor(_root_.scalapb.grpc.ConcreteProtoMethodDescriptorSupplier.fromMethodDescriptor(pisa.server.ServerProto.javaDescriptor.getServices.get(0).getMethods.get(1)))
      .build()
  
  val METHOD_ISABELLE_WORKING_DIRECTORY: _root_.io.grpc.MethodDescriptor[pisa.server.IsaPath, pisa.server.IsaMessage] =
    _root_.io.grpc.MethodDescriptor.newBuilder()
      .setType(_root_.io.grpc.MethodDescriptor.MethodType.UNARY)
      .setFullMethodName(_root_.io.grpc.MethodDescriptor.generateFullMethodName("server.Server", "IsabelleWorkingDirectory"))
      .setSampledToLocalTracing(true)
      .setRequestMarshaller(_root_.scalapb.grpc.Marshaller.forMessage[pisa.server.IsaPath])
      .setResponseMarshaller(_root_.scalapb.grpc.Marshaller.forMessage[pisa.server.IsaMessage])
      .setSchemaDescriptor(_root_.scalapb.grpc.ConcreteProtoMethodDescriptorSupplier.fromMethodDescriptor(pisa.server.ServerProto.javaDescriptor.getServices.get(0).getMethods.get(2)))
      .build()
  
  val METHOD_ISABELLE_COMMAND: _root_.io.grpc.MethodDescriptor[pisa.server.IsaCommand, pisa.server.IsaState] =
    _root_.io.grpc.MethodDescriptor.newBuilder()
      .setType(_root_.io.grpc.MethodDescriptor.MethodType.UNARY)
      .setFullMethodName(_root_.io.grpc.MethodDescriptor.generateFullMethodName("server.Server", "IsabelleCommand"))
      .setSampledToLocalTracing(true)
      .setRequestMarshaller(_root_.scalapb.grpc.Marshaller.forMessage[pisa.server.IsaCommand])
      .setResponseMarshaller(_root_.scalapb.grpc.Marshaller.forMessage[pisa.server.IsaState])
      .setSchemaDescriptor(_root_.scalapb.grpc.ConcreteProtoMethodDescriptorSupplier.fromMethodDescriptor(pisa.server.ServerProto.javaDescriptor.getServices.get(0).getMethods.get(3)))
      .build()
  
  val METHOD_ISABELLE_SET_SEARCH_WIDTH: _root_.io.grpc.MethodDescriptor[pisa.server.IsaSearchWidth, pisa.server.IsaMessage] =
    _root_.io.grpc.MethodDescriptor.newBuilder()
      .setType(_root_.io.grpc.MethodDescriptor.MethodType.UNARY)
      .setFullMethodName(_root_.io.grpc.MethodDescriptor.generateFullMethodName("server.Server", "IsabelleSetSearchWidth"))
      .setSampledToLocalTracing(true)
      .setRequestMarshaller(_root_.scalapb.grpc.Marshaller.forMessage[pisa.server.IsaSearchWidth])
      .setResponseMarshaller(_root_.scalapb.grpc.Marshaller.forMessage[pisa.server.IsaMessage])
      .setSchemaDescriptor(_root_.scalapb.grpc.ConcreteProtoMethodDescriptorSupplier.fromMethodDescriptor(pisa.server.ServerProto.javaDescriptor.getServices.get(0).getMethods.get(4)))
      .build()
  
  val METHOD_ISABELLE_SEARCH_INDEX_COMMAND: _root_.io.grpc.MethodDescriptor[pisa.server.IsaSearchIndexCommand, pisa.server.IsaState] =
    _root_.io.grpc.MethodDescriptor.newBuilder()
      .setType(_root_.io.grpc.MethodDescriptor.MethodType.UNARY)
      .setFullMethodName(_root_.io.grpc.MethodDescriptor.generateFullMethodName("server.Server", "IsabelleSearchIndexCommand"))
      .setSampledToLocalTracing(true)
      .setRequestMarshaller(_root_.scalapb.grpc.Marshaller.forMessage[pisa.server.IsaSearchIndexCommand])
      .setResponseMarshaller(_root_.scalapb.grpc.Marshaller.forMessage[pisa.server.IsaState])
      .setSchemaDescriptor(_root_.scalapb.grpc.ConcreteProtoMethodDescriptorSupplier.fromMethodDescriptor(pisa.server.ServerProto.javaDescriptor.getServices.get(0).getMethods.get(5)))
      .build()
  
  val SERVICE: _root_.io.grpc.ServiceDescriptor =
    _root_.io.grpc.ServiceDescriptor.newBuilder("server.Server")
      .setSchemaDescriptor(new _root_.scalapb.grpc.ConcreteProtoFileDescriptorSupplier(pisa.server.ServerProto.javaDescriptor))
      .addMethod(METHOD_INITIALISE_ISABELLE)
      .addMethod(METHOD_ISABELLE_CONTEXT)
      .addMethod(METHOD_ISABELLE_WORKING_DIRECTORY)
      .addMethod(METHOD_ISABELLE_COMMAND)
      .addMethod(METHOD_ISABELLE_SET_SEARCH_WIDTH)
      .addMethod(METHOD_ISABELLE_SEARCH_INDEX_COMMAND)
      .build()
  
  trait Server extends _root_.scalapb.grpc.AbstractService {
    override def serviceCompanion = Server
    def initialiseIsabelle(request: pisa.server.IsaPath): scala.concurrent.Future[pisa.server.IsaMessage]
    def isabelleContext(request: pisa.server.IsaContext): scala.concurrent.Future[pisa.server.IsaMessage]
    def isabelleWorkingDirectory(request: pisa.server.IsaPath): scala.concurrent.Future[pisa.server.IsaMessage]
    def isabelleCommand(request: pisa.server.IsaCommand): scala.concurrent.Future[pisa.server.IsaState]
    def isabelleSetSearchWidth(request: pisa.server.IsaSearchWidth): scala.concurrent.Future[pisa.server.IsaMessage]
    def isabelleSearchIndexCommand(request: pisa.server.IsaSearchIndexCommand): scala.concurrent.Future[pisa.server.IsaState]
  }
  
  object Server extends _root_.scalapb.grpc.ServiceCompanion[Server] {
    implicit def serviceCompanion: _root_.scalapb.grpc.ServiceCompanion[Server] = this
    def javaDescriptor: _root_.com.google.protobuf.Descriptors.ServiceDescriptor = pisa.server.ServerProto.javaDescriptor.getServices.get(0)
    def scalaDescriptor: _root_.scalapb.descriptors.ServiceDescriptor = pisa.server.ServerProto.scalaDescriptor.services(0)
    def bindService(serviceImpl: Server, executionContext: scala.concurrent.ExecutionContext): _root_.io.grpc.ServerServiceDefinition =
      _root_.io.grpc.ServerServiceDefinition.builder(SERVICE)
      .addMethod(
        METHOD_INITIALISE_ISABELLE,
        _root_.io.grpc.stub.ServerCalls.asyncUnaryCall(new _root_.io.grpc.stub.ServerCalls.UnaryMethod[pisa.server.IsaPath, pisa.server.IsaMessage] {
          override def invoke(request: pisa.server.IsaPath, observer: _root_.io.grpc.stub.StreamObserver[pisa.server.IsaMessage]): Unit =
            serviceImpl.initialiseIsabelle(request).onComplete(scalapb.grpc.Grpc.completeObserver(observer))(
              executionContext)
        }))
      .addMethod(
        METHOD_ISABELLE_CONTEXT,
        _root_.io.grpc.stub.ServerCalls.asyncUnaryCall(new _root_.io.grpc.stub.ServerCalls.UnaryMethod[pisa.server.IsaContext, pisa.server.IsaMessage] {
          override def invoke(request: pisa.server.IsaContext, observer: _root_.io.grpc.stub.StreamObserver[pisa.server.IsaMessage]): Unit =
            serviceImpl.isabelleContext(request).onComplete(scalapb.grpc.Grpc.completeObserver(observer))(
              executionContext)
        }))
      .addMethod(
        METHOD_ISABELLE_WORKING_DIRECTORY,
        _root_.io.grpc.stub.ServerCalls.asyncUnaryCall(new _root_.io.grpc.stub.ServerCalls.UnaryMethod[pisa.server.IsaPath, pisa.server.IsaMessage] {
          override def invoke(request: pisa.server.IsaPath, observer: _root_.io.grpc.stub.StreamObserver[pisa.server.IsaMessage]): Unit =
            serviceImpl.isabelleWorkingDirectory(request).onComplete(scalapb.grpc.Grpc.completeObserver(observer))(
              executionContext)
        }))
      .addMethod(
        METHOD_ISABELLE_COMMAND,
        _root_.io.grpc.stub.ServerCalls.asyncUnaryCall(new _root_.io.grpc.stub.ServerCalls.UnaryMethod[pisa.server.IsaCommand, pisa.server.IsaState] {
          override def invoke(request: pisa.server.IsaCommand, observer: _root_.io.grpc.stub.StreamObserver[pisa.server.IsaState]): Unit =
            serviceImpl.isabelleCommand(request).onComplete(scalapb.grpc.Grpc.completeObserver(observer))(
              executionContext)
        }))
      .addMethod(
        METHOD_ISABELLE_SET_SEARCH_WIDTH,
        _root_.io.grpc.stub.ServerCalls.asyncUnaryCall(new _root_.io.grpc.stub.ServerCalls.UnaryMethod[pisa.server.IsaSearchWidth, pisa.server.IsaMessage] {
          override def invoke(request: pisa.server.IsaSearchWidth, observer: _root_.io.grpc.stub.StreamObserver[pisa.server.IsaMessage]): Unit =
            serviceImpl.isabelleSetSearchWidth(request).onComplete(scalapb.grpc.Grpc.completeObserver(observer))(
              executionContext)
        }))
      .addMethod(
        METHOD_ISABELLE_SEARCH_INDEX_COMMAND,
        _root_.io.grpc.stub.ServerCalls.asyncUnaryCall(new _root_.io.grpc.stub.ServerCalls.UnaryMethod[pisa.server.IsaSearchIndexCommand, pisa.server.IsaState] {
          override def invoke(request: pisa.server.IsaSearchIndexCommand, observer: _root_.io.grpc.stub.StreamObserver[pisa.server.IsaState]): Unit =
            serviceImpl.isabelleSearchIndexCommand(request).onComplete(scalapb.grpc.Grpc.completeObserver(observer))(
              executionContext)
        }))
      .build()
  }
  
  trait ServerBlockingClient {
    def serviceCompanion = Server
    def initialiseIsabelle(request: pisa.server.IsaPath): pisa.server.IsaMessage
    def isabelleContext(request: pisa.server.IsaContext): pisa.server.IsaMessage
    def isabelleWorkingDirectory(request: pisa.server.IsaPath): pisa.server.IsaMessage
    def isabelleCommand(request: pisa.server.IsaCommand): pisa.server.IsaState
    def isabelleSetSearchWidth(request: pisa.server.IsaSearchWidth): pisa.server.IsaMessage
    def isabelleSearchIndexCommand(request: pisa.server.IsaSearchIndexCommand): pisa.server.IsaState
  }
  
  class ServerBlockingStub(channel: _root_.io.grpc.Channel, options: _root_.io.grpc.CallOptions = _root_.io.grpc.CallOptions.DEFAULT) extends _root_.io.grpc.stub.AbstractStub[ServerBlockingStub](channel, options) with ServerBlockingClient {
    override def initialiseIsabelle(request: pisa.server.IsaPath): pisa.server.IsaMessage = {
      _root_.scalapb.grpc.ClientCalls.blockingUnaryCall(channel, METHOD_INITIALISE_ISABELLE, options, request)
    }
    
    override def isabelleContext(request: pisa.server.IsaContext): pisa.server.IsaMessage = {
      _root_.scalapb.grpc.ClientCalls.blockingUnaryCall(channel, METHOD_ISABELLE_CONTEXT, options, request)
    }
    
    override def isabelleWorkingDirectory(request: pisa.server.IsaPath): pisa.server.IsaMessage = {
      _root_.scalapb.grpc.ClientCalls.blockingUnaryCall(channel, METHOD_ISABELLE_WORKING_DIRECTORY, options, request)
    }
    
    override def isabelleCommand(request: pisa.server.IsaCommand): pisa.server.IsaState = {
      _root_.scalapb.grpc.ClientCalls.blockingUnaryCall(channel, METHOD_ISABELLE_COMMAND, options, request)
    }
    
    override def isabelleSetSearchWidth(request: pisa.server.IsaSearchWidth): pisa.server.IsaMessage = {
      _root_.scalapb.grpc.ClientCalls.blockingUnaryCall(channel, METHOD_ISABELLE_SET_SEARCH_WIDTH, options, request)
    }
    
    override def isabelleSearchIndexCommand(request: pisa.server.IsaSearchIndexCommand): pisa.server.IsaState = {
      _root_.scalapb.grpc.ClientCalls.blockingUnaryCall(channel, METHOD_ISABELLE_SEARCH_INDEX_COMMAND, options, request)
    }
    
    override def build(channel: _root_.io.grpc.Channel, options: _root_.io.grpc.CallOptions): ServerBlockingStub = new ServerBlockingStub(channel, options)
  }
  
  class ServerStub(channel: _root_.io.grpc.Channel, options: _root_.io.grpc.CallOptions = _root_.io.grpc.CallOptions.DEFAULT) extends _root_.io.grpc.stub.AbstractStub[ServerStub](channel, options) with Server {
    override def initialiseIsabelle(request: pisa.server.IsaPath): scala.concurrent.Future[pisa.server.IsaMessage] = {
      _root_.scalapb.grpc.ClientCalls.asyncUnaryCall(channel, METHOD_INITIALISE_ISABELLE, options, request)
    }
    
    override def isabelleContext(request: pisa.server.IsaContext): scala.concurrent.Future[pisa.server.IsaMessage] = {
      _root_.scalapb.grpc.ClientCalls.asyncUnaryCall(channel, METHOD_ISABELLE_CONTEXT, options, request)
    }
    
    override def isabelleWorkingDirectory(request: pisa.server.IsaPath): scala.concurrent.Future[pisa.server.IsaMessage] = {
      _root_.scalapb.grpc.ClientCalls.asyncUnaryCall(channel, METHOD_ISABELLE_WORKING_DIRECTORY, options, request)
    }
    
    override def isabelleCommand(request: pisa.server.IsaCommand): scala.concurrent.Future[pisa.server.IsaState] = {
      _root_.scalapb.grpc.ClientCalls.asyncUnaryCall(channel, METHOD_ISABELLE_COMMAND, options, request)
    }
    
    override def isabelleSetSearchWidth(request: pisa.server.IsaSearchWidth): scala.concurrent.Future[pisa.server.IsaMessage] = {
      _root_.scalapb.grpc.ClientCalls.asyncUnaryCall(channel, METHOD_ISABELLE_SET_SEARCH_WIDTH, options, request)
    }
    
    override def isabelleSearchIndexCommand(request: pisa.server.IsaSearchIndexCommand): scala.concurrent.Future[pisa.server.IsaState] = {
      _root_.scalapb.grpc.ClientCalls.asyncUnaryCall(channel, METHOD_ISABELLE_SEARCH_INDEX_COMMAND, options, request)
    }
    
    override def build(channel: _root_.io.grpc.Channel, options: _root_.io.grpc.CallOptions): ServerStub = new ServerStub(channel, options)
  }
  
  def bindService(serviceImpl: Server, executionContext: scala.concurrent.ExecutionContext): _root_.io.grpc.ServerServiceDefinition = Server.bindService(serviceImpl, executionContext)
  
  def blockingStub(channel: _root_.io.grpc.Channel): ServerBlockingStub = new ServerBlockingStub(channel)
  
  def stub(channel: _root_.io.grpc.Channel): ServerStub = new ServerStub(channel)
  
  def javaDescriptor: _root_.com.google.protobuf.Descriptors.ServiceDescriptor = pisa.server.ServerProto.javaDescriptor.getServices.get(0)
  
}