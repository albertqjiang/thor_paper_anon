package pisa.server

import scala.language.implicitConversions

object ZioServer {
  trait ZServer[-R, -Context] extends scalapb.zio_grpc.ZGeneratedService[R, Context, ZServer] {
    self =>
    def initialiseIsabelle(request: pisa.server.IsaPath): _root_.zio.ZIO[R with Context, io.grpc.Status, pisa.server.IsaMessage]
    def isabelleContext(request: pisa.server.IsaContext): _root_.zio.ZIO[R with Context, io.grpc.Status, pisa.server.IsaMessage]
    def isabelleWorkingDirectory(request: pisa.server.IsaPath): _root_.zio.ZIO[R with Context, io.grpc.Status, pisa.server.IsaMessage]
    def isabelleCommand(request: pisa.server.IsaCommand): _root_.zio.ZIO[R with Context, io.grpc.Status, pisa.server.IsaState]
    def isabelleSetSearchWidth(request: pisa.server.IsaSearchWidth): _root_.zio.ZIO[R with Context, io.grpc.Status, pisa.server.IsaMessage]
    def isabelleSearchIndexCommand(request: pisa.server.IsaSearchIndexCommand): _root_.zio.ZIO[R with Context, io.grpc.Status, pisa.server.IsaState]
  }
  type Server = ZServer[Any, Any]
  type RServer[R] = ZServer[R, Any]
  type RCServer[R] = ZServer[R, zio.Has[scalapb.zio_grpc.RequestContext]]
  
  object ZServer {
    implicit val transformableService: scalapb.zio_grpc.TransformableService[ZServer] = new scalapb.zio_grpc.TransformableService[ZServer] {
      def transform[R, Context, R1, Context1](self: ZServer[R, Context], f: scalapb.zio_grpc.ZTransform[R with Context, io.grpc.Status, R1 with Context1]): pisa.server.ZioServer.ZServer[R1, Context1] = new pisa.server.ZioServer.ZServer[R1, Context1] {
        def initialiseIsabelle(request: pisa.server.IsaPath): _root_.zio.ZIO[R1 with Context1, io.grpc.Status, pisa.server.IsaMessage] = f.effect(self.initialiseIsabelle(request))
        def isabelleContext(request: pisa.server.IsaContext): _root_.zio.ZIO[R1 with Context1, io.grpc.Status, pisa.server.IsaMessage] = f.effect(self.isabelleContext(request))
        def isabelleWorkingDirectory(request: pisa.server.IsaPath): _root_.zio.ZIO[R1 with Context1, io.grpc.Status, pisa.server.IsaMessage] = f.effect(self.isabelleWorkingDirectory(request))
        def isabelleCommand(request: pisa.server.IsaCommand): _root_.zio.ZIO[R1 with Context1, io.grpc.Status, pisa.server.IsaState] = f.effect(self.isabelleCommand(request))
        def isabelleSetSearchWidth(request: pisa.server.IsaSearchWidth): _root_.zio.ZIO[R1 with Context1, io.grpc.Status, pisa.server.IsaMessage] = f.effect(self.isabelleSetSearchWidth(request))
        def isabelleSearchIndexCommand(request: pisa.server.IsaSearchIndexCommand): _root_.zio.ZIO[R1 with Context1, io.grpc.Status, pisa.server.IsaState] = f.effect(self.isabelleSearchIndexCommand(request))
      }
    }
    implicit def ops[R, C](service: pisa.server.ZioServer.ZServer[R, C]) = new scalapb.zio_grpc.TransformableService.TransformableServiceOps[pisa.server.ZioServer.ZServer, R, C](service)
    implicit val genericBindable: scalapb.zio_grpc.GenericBindable[pisa.server.ZioServer.ZServer] = new scalapb.zio_grpc.GenericBindable[pisa.server.ZioServer.ZServer] {
      def bind[R, C](serviceImpl: pisa.server.ZioServer.ZServer[R, C], env: zio.Has[scalapb.zio_grpc.RequestContext] => R with C): zio.URIO[R, _root_.io.grpc.ServerServiceDefinition] =
        zio.ZIO.runtime[Any].map {
          runtime =>
            _root_.io.grpc.ServerServiceDefinition.builder(pisa.server.ServerGrpc.SERVICE)
            .addMethod(
              pisa.server.ServerGrpc.METHOD_INITIALISE_ISABELLE,
              _root_.scalapb.zio_grpc.server.ZServerCallHandler.unaryCallHandler(runtime, (t: pisa.server.IsaPath)=>serviceImpl.initialiseIsabelle(t).provideSome(env))
            )
            .addMethod(
              pisa.server.ServerGrpc.METHOD_ISABELLE_CONTEXT,
              _root_.scalapb.zio_grpc.server.ZServerCallHandler.unaryCallHandler(runtime, (t: pisa.server.IsaContext)=>serviceImpl.isabelleContext(t).provideSome(env))
            )
            .addMethod(
              pisa.server.ServerGrpc.METHOD_ISABELLE_WORKING_DIRECTORY,
              _root_.scalapb.zio_grpc.server.ZServerCallHandler.unaryCallHandler(runtime, (t: pisa.server.IsaPath)=>serviceImpl.isabelleWorkingDirectory(t).provideSome(env))
            )
            .addMethod(
              pisa.server.ServerGrpc.METHOD_ISABELLE_COMMAND,
              _root_.scalapb.zio_grpc.server.ZServerCallHandler.unaryCallHandler(runtime, (t: pisa.server.IsaCommand)=>serviceImpl.isabelleCommand(t).provideSome(env))
            )
            .addMethod(
              pisa.server.ServerGrpc.METHOD_ISABELLE_SET_SEARCH_WIDTH,
              _root_.scalapb.zio_grpc.server.ZServerCallHandler.unaryCallHandler(runtime, (t: pisa.server.IsaSearchWidth)=>serviceImpl.isabelleSetSearchWidth(t).provideSome(env))
            )
            .addMethod(
              pisa.server.ServerGrpc.METHOD_ISABELLE_SEARCH_INDEX_COMMAND,
              _root_.scalapb.zio_grpc.server.ZServerCallHandler.unaryCallHandler(runtime, (t: pisa.server.IsaSearchIndexCommand)=>serviceImpl.isabelleSearchIndexCommand(t).provideSome(env))
            )
            .build()
        }
      }
  }
  
  type ServerClient = _root_.zio.Has[ServerClient.Service]
  
  // accessor methods
  class ServerAccessors[Context: zio.Tag](callOptions: zio.IO[io.grpc.Status, io.grpc.CallOptions]) extends scalapb.zio_grpc.CallOptionsMethods[ServerAccessors[Context]] {
    def this() = this(zio.ZIO.succeed(io.grpc.CallOptions.DEFAULT))
    def initialiseIsabelle(request: pisa.server.IsaPath): _root_.zio.ZIO[zio.Has[ServerClient.ZService[Any, Context]] with Context, io.grpc.Status, pisa.server.IsaMessage] = _root_.zio.ZIO.accessM(_.get.withCallOptionsM(callOptions).initialiseIsabelle(request))
    def isabelleContext(request: pisa.server.IsaContext): _root_.zio.ZIO[zio.Has[ServerClient.ZService[Any, Context]] with Context, io.grpc.Status, pisa.server.IsaMessage] = _root_.zio.ZIO.accessM(_.get.withCallOptionsM(callOptions).isabelleContext(request))
    def isabelleWorkingDirectory(request: pisa.server.IsaPath): _root_.zio.ZIO[zio.Has[ServerClient.ZService[Any, Context]] with Context, io.grpc.Status, pisa.server.IsaMessage] = _root_.zio.ZIO.accessM(_.get.withCallOptionsM(callOptions).isabelleWorkingDirectory(request))
    def isabelleCommand(request: pisa.server.IsaCommand): _root_.zio.ZIO[zio.Has[ServerClient.ZService[Any, Context]] with Context, io.grpc.Status, pisa.server.IsaState] = _root_.zio.ZIO.accessM(_.get.withCallOptionsM(callOptions).isabelleCommand(request))
    def isabelleSetSearchWidth(request: pisa.server.IsaSearchWidth): _root_.zio.ZIO[zio.Has[ServerClient.ZService[Any, Context]] with Context, io.grpc.Status, pisa.server.IsaMessage] = _root_.zio.ZIO.accessM(_.get.withCallOptionsM(callOptions).isabelleSetSearchWidth(request))
    def isabelleSearchIndexCommand(request: pisa.server.IsaSearchIndexCommand): _root_.zio.ZIO[zio.Has[ServerClient.ZService[Any, Context]] with Context, io.grpc.Status, pisa.server.IsaState] = _root_.zio.ZIO.accessM(_.get.withCallOptionsM(callOptions).isabelleSearchIndexCommand(request))
    def mapCallOptionsM(f: io.grpc.CallOptions => zio.IO[io.grpc.Status, io.grpc.CallOptions]) = new ServerAccessors(callOptions.flatMap(f))
  }
  
  object ServerClient extends ServerAccessors[Any](zio.ZIO.succeed(io.grpc.CallOptions.DEFAULT)) {
    trait ZService[R, Context] extends scalapb.zio_grpc.CallOptionsMethods[ZService[R, Context]] {
      def initialiseIsabelle(request: pisa.server.IsaPath): _root_.zio.ZIO[R with Context, io.grpc.Status, pisa.server.IsaMessage]
      def isabelleContext(request: pisa.server.IsaContext): _root_.zio.ZIO[R with Context, io.grpc.Status, pisa.server.IsaMessage]
      def isabelleWorkingDirectory(request: pisa.server.IsaPath): _root_.zio.ZIO[R with Context, io.grpc.Status, pisa.server.IsaMessage]
      def isabelleCommand(request: pisa.server.IsaCommand): _root_.zio.ZIO[R with Context, io.grpc.Status, pisa.server.IsaState]
      def isabelleSetSearchWidth(request: pisa.server.IsaSearchWidth): _root_.zio.ZIO[R with Context, io.grpc.Status, pisa.server.IsaMessage]
      def isabelleSearchIndexCommand(request: pisa.server.IsaSearchIndexCommand): _root_.zio.ZIO[R with Context, io.grpc.Status, pisa.server.IsaState]
      
      // Returns a copy of the service with new default metadata
      def withMetadataM[C](headersEffect: zio.ZIO[C, io.grpc.Status, scalapb.zio_grpc.SafeMetadata]): ZService[R, C]
      def withCallOptionsM(callOptions: zio.IO[io.grpc.Status, io.grpc.CallOptions]): ZService[R, Context]
      def withMetadata(headers: scalapb.zio_grpc.SafeMetadata): ZService[R, Any] = withMetadataM(zio.ZIO.succeed(headers))
    }
    type Service = ZService[Any, Any]
    type Accessors[Context] = pisa.server.ZioServer.ServerAccessors[Context]
    
    
    private[this] class ServiceStub[R, Context](channel: scalapb.zio_grpc.ZChannel[R], options: zio.IO[io.grpc.Status, io.grpc.CallOptions], headers: zio.ZIO[Context, io.grpc.Status, scalapb.zio_grpc.SafeMetadata])
        extends ServerClient.ZService[R, Context] {
      def initialiseIsabelle(request: pisa.server.IsaPath): _root_.zio.ZIO[R with Context, io.grpc.Status, pisa.server.IsaMessage] = headers.zip(options).flatMap { case (headers, options) => scalapb.zio_grpc.client.ClientCalls.unaryCall(
        channel, pisa.server.ServerGrpc.METHOD_INITIALISE_ISABELLE, options,
        headers,
        request
      )}
      def isabelleContext(request: pisa.server.IsaContext): _root_.zio.ZIO[R with Context, io.grpc.Status, pisa.server.IsaMessage] = headers.zip(options).flatMap { case (headers, options) => scalapb.zio_grpc.client.ClientCalls.unaryCall(
        channel, pisa.server.ServerGrpc.METHOD_ISABELLE_CONTEXT, options,
        headers,
        request
      )}
      def isabelleWorkingDirectory(request: pisa.server.IsaPath): _root_.zio.ZIO[R with Context, io.grpc.Status, pisa.server.IsaMessage] = headers.zip(options).flatMap { case (headers, options) => scalapb.zio_grpc.client.ClientCalls.unaryCall(
        channel, pisa.server.ServerGrpc.METHOD_ISABELLE_WORKING_DIRECTORY, options,
        headers,
        request
      )}
      def isabelleCommand(request: pisa.server.IsaCommand): _root_.zio.ZIO[R with Context, io.grpc.Status, pisa.server.IsaState] = headers.zip(options).flatMap { case (headers, options) => scalapb.zio_grpc.client.ClientCalls.unaryCall(
        channel, pisa.server.ServerGrpc.METHOD_ISABELLE_COMMAND, options,
        headers,
        request
      )}
      def isabelleSetSearchWidth(request: pisa.server.IsaSearchWidth): _root_.zio.ZIO[R with Context, io.grpc.Status, pisa.server.IsaMessage] = headers.zip(options).flatMap { case (headers, options) => scalapb.zio_grpc.client.ClientCalls.unaryCall(
        channel, pisa.server.ServerGrpc.METHOD_ISABELLE_SET_SEARCH_WIDTH, options,
        headers,
        request
      )}
      def isabelleSearchIndexCommand(request: pisa.server.IsaSearchIndexCommand): _root_.zio.ZIO[R with Context, io.grpc.Status, pisa.server.IsaState] = headers.zip(options).flatMap { case (headers, options) => scalapb.zio_grpc.client.ClientCalls.unaryCall(
        channel, pisa.server.ServerGrpc.METHOD_ISABELLE_SEARCH_INDEX_COMMAND, options,
        headers,
        request
      )}
      def mapCallOptionsM(f: io.grpc.CallOptions => zio.IO[io.grpc.Status, io.grpc.CallOptions]): ZService[R, Context] = new ServiceStub(channel, options.flatMap(f), headers)
      override def withMetadataM[C](headersEffect: zio.ZIO[C, io.grpc.Status, scalapb.zio_grpc.SafeMetadata]): ZService[R, C] = new ServiceStub(channel, options, headersEffect)
      def withCallOptionsM(callOptions: zio.IO[io.grpc.Status, io.grpc.CallOptions]): ZService[R, Context] = new ServiceStub(channel, callOptions, headers)
    }
    
    def managed[R, Context](managedChannel: scalapb.zio_grpc.ZManagedChannel[R], options: zio.IO[io.grpc.Status, io.grpc.CallOptions] = zio.ZIO.succeed(io.grpc.CallOptions.DEFAULT), headers: zio.ZIO[Context, io.grpc.Status, scalapb.zio_grpc.SafeMetadata]=scalapb.zio_grpc.SafeMetadata.make): zio.Managed[Throwable, ServerClient.ZService[R, Context]] = managedChannel.map {
      channel => new ServiceStub[R, Context](channel, options, headers)
    }
    
    def live[R](managedChannel: scalapb.zio_grpc.ZManagedChannel[R]): zio.ZLayer[R, Throwable, ServerClient] = live(managedChannel, zio.ZIO.succeed(io.grpc.CallOptions.DEFAULT), scalapb.zio_grpc.SafeMetadata.make)
    def live[R, Context: zio.Tag](managedChannel: scalapb.zio_grpc.ZManagedChannel[R], options: zio.IO[io.grpc.Status, io.grpc.CallOptions] = zio.ZIO.succeed(io.grpc.CallOptions.DEFAULT), headers: zio.ZIO[Context, io.grpc.Status, scalapb.zio_grpc.SafeMetadata]): zio.ZLayer[R, Throwable, zio.Has[ServerClient.ZService[Any, Context]]] = zio.ZLayer.fromFunctionManaged((r: R) => managed[Any, Context](managedChannel.map(_.provide(r)), options, headers))
  }
}