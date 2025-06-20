# src/mcp_server/main.py

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

from .config import MCPConfig
from .server import MCPServer
from .session_manager import SessionManager
from .monitoring import MetricsCollector, create_default_health_checks
from .error_handlers import log_error, InternalError
from .tool_registry import ToolRegistry
from .service_factory import ServiceFactory

# Import all tool categories
from .tools.embedding_tools import EmbeddingGenerationTool, BatchEmbeddingTool, MultimodalEmbeddingTool
from .tools.search_tools import SemanticSearchTool, SimilaritySearchTool, FacetedSearchTool
from .tools.storage_tools import StorageManagementTool, CollectionManagementTool, RetrievalTool
from .tools.analysis_tools import ClusterAnalysisTool, QualityAssessmentTool, DimensionalityReductionTool
from .tools.vector_store_tools import VectorIndexTool, VectorRetrievalTool, VectorMetadataTool
from .tools.ipfs_cluster_tools import IPFSClusterTool, DistributedVectorTool, IPFSMetadataTool

logger = logging.getLogger(__name__)

class MCPServerApplication:
    """
    Main MCP Server application that orchestrates all components.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        # Load configuration
        if config_file and Path(config_file).exists():
            self.config = MCPConfig.from_env_file(config_file)
        else:
            self.config = MCPConfig()
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize components
        self.service_factory: Optional[ServiceFactory] = None
        self.session_manager: Optional[SessionManager] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.tool_registry: Optional[ToolRegistry] = None
        self.mcp_server: Optional[MCPServer] = None
        
        # Shutdown event
        self.shutdown_event = asyncio.Event()
        
        logger.info(f"Initializing MCP Server: {self.config.server_name} v{self.config.server_version}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(self.config.log_format)
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler if specified
        if self.config.log_file:
            try:
                file_handler = logging.FileHandler(self.config.log_file)
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
                logger.info(f"Logging to file: {self.config.log_file}")
            except Exception as e:
                logger.warning(f"Failed to setup file logging: {e}")
    
    async def _initialize_components(self):
        """Initialize all server components."""
        try:
            # Initialize service factory first
            logger.info("Initializing service factory...")
            self.service_factory = ServiceFactory(self.config)
            await self.service_factory.initialize_services()
            
            # Initialize session manager
            logger.info("Initializing session manager...")
            self.session_manager = SessionManager(self.config)
            
            # Initialize metrics collector
            logger.info("Initializing metrics collector...")
            self.metrics_collector = MetricsCollector(self.config)
            
            # Add default health checks
            if self.metrics_collector.enabled:
                health_checks = create_default_health_checks(self.config)
                for check_func in health_checks:
                    try:
                        result = check_func()
                        self.metrics_collector.health_checks[result.component] = result
                    except Exception as e:
                        logger.warning(f"Failed to initialize health check: {e}")
            
            # Initialize tool registry
            logger.info("Initializing tool registry...")
            self.tool_registry = ToolRegistry()
            
            # Register all tools with real service instances
            await self._register_tools()
            
            # Initialize MCP server
            logger.info("Initializing MCP server...")
            self.mcp_server = MCPServer(
                name=self.config.server_name,
                version=self.config.server_version,
                session_manager=self.session_manager,
                metrics_collector=self.metrics_collector,
                tool_registry=self.tool_registry,
                config=self.config
            )
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            log_error(InternalError(f"Component initialization failed: {e}"))
            raise
    
    async def _register_tools(self):
        """Register all available tools with real service instances."""
        if not self.tool_registry:
            raise InternalError("Tool registry not initialized")
        
        if not self.service_factory:
            raise InternalError("Service factory not initialized")
        
        try:
            # Get service instances from factory
            embedding_service = self.service_factory.get_embedding_service()
            vector_service = self.service_factory.get_vector_service()
            clustering_service = self.service_factory.get_clustering_service()
            ipfs_vector_service = self.service_factory.get_ipfs_vector_service()
            distributed_vector_service = self.service_factory.get_distributed_vector_service()
            
            # Register embedding tools
            logger.info("Registering embedding tools...")
            embedding_gen = EmbeddingGenerationTool(embedding_service)
            batch_embedding = BatchEmbeddingTool(embedding_service)
            multimodal_embedding = MultimodalEmbeddingTool(embedding_service)
            
            self.tool_registry.register_tool(embedding_gen)
            self.tool_registry.register_tool(batch_embedding)
            self.tool_registry.register_tool(multimodal_embedding)
            
            # Register search tools
            logger.info("Registering search tools...")
            semantic_search = SemanticSearchTool(vector_service, embedding_service)
            similarity_search = SimilaritySearchTool(vector_service)
            faceted_search = FacetedSearchTool(vector_service)
            
            self.tool_registry.register_tool(semantic_search)
            self.tool_registry.register_tool(similarity_search)
            self.tool_registry.register_tool(faceted_search)
            
            # Register storage tools
            logger.info("Registering storage tools...")
            storage_mgmt = StorageManagementTool(vector_service)
            collection_mgmt = CollectionManagementTool(vector_service)
            retrieval = RetrievalTool(vector_service)
            
            self.tool_registry.register_tool(storage_mgmt)
            self.tool_registry.register_tool(collection_mgmt)
            self.tool_registry.register_tool(retrieval)
            
            # Register analysis tools
            logger.info("Registering analysis tools...")
            cluster_analysis = ClusterAnalysisTool(clustering_service)
            quality_assessment = QualityAssessmentTool(vector_service)
            dimensionality_reduction = DimensionalityReductionTool(vector_service)
            
            self.tool_registry.register_tool(cluster_analysis)
            self.tool_registry.register_tool(quality_assessment)
            self.tool_registry.register_tool(dimensionality_reduction)
            
            # Register vector store tools
            logger.info("Registering vector store tools...")
            vector_index = VectorIndexTool(vector_service)
            vector_retrieval = VectorRetrievalTool(vector_service)
            vector_metadata = VectorMetadataTool(vector_service)
            
            self.tool_registry.register_tool(vector_index)
            self.tool_registry.register_tool(vector_retrieval)
            self.tool_registry.register_tool(vector_metadata)
            
            # Register IPFS cluster tools (optional)
            logger.info("Registering IPFS cluster tools...")
            ipfs_cluster = IPFSClusterTool(ipfs_vector_service)
            distributed_vector = DistributedVectorTool(distributed_vector_service)
            ipfs_metadata = IPFSMetadataTool(ipfs_vector_service)
            
            self.tool_registry.register_tool(ipfs_cluster)
            self.tool_registry.register_tool(distributed_vector)
            self.tool_registry.register_tool(ipfs_metadata)
            
            total_tools = len(self.tool_registry.get_all_tools())
            logger.info(f"Successfully registered {total_tools} tools")
            
        except Exception as e:
            logger.error(f"Failed to register tools: {e}")
            log_error(InternalError(f"Tool registration failed: {e}"))
            raise
    
    def get_tools(self):
        """Get all registered tools."""
        if self.tool_registry:
            return self.tool_registry.get_all_tools()
        return {}
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _shutdown_components(self):
        """Shutdown all components gracefully."""
        logger.info("Starting graceful shutdown...")
        
        try:
            # Stop MCP server
            if self.mcp_server:
                logger.info("Stopping MCP server...")
                await self.mcp_server.stop()
            
            # Shutdown metrics collector
            if self.metrics_collector:
                logger.info("Stopping metrics collector...")
                await self.metrics_collector.shutdown()
            
            # Shutdown session manager
            if self.session_manager:
                logger.info("Stopping session manager...")
                await self.session_manager.shutdown()
            
            # Shutdown service factory
            if self.service_factory:
                logger.info("Stopping service factory...")
                await self.service_factory.shutdown()
                
            logger.info("Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def run(self):
        """Run the MCP server application."""
        try:
            # Setup signal handlers
            self._setup_signal_handlers()
            
            # Initialize all components
            await self._initialize_components()
            
            # Start the server
            logger.info(f"Starting MCP server on stdio...")
            
            if self.mcp_server:
                # Run the server and wait for shutdown signal
                server_task = asyncio.create_task(self.mcp_server.run())
                shutdown_task = asyncio.create_task(self.shutdown_event.wait())
                
                # Wait for either server completion or shutdown signal
                done, pending = await asyncio.wait(
                    [server_task, shutdown_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
                # Check if server task completed with error
                if server_task in done:
                    try:
                        await server_task
                    except Exception as e:
                        logger.error(f"Server task failed: {e}")
                        raise
            
        except Exception as e:
            logger.error(f"Failed to run server: {e}")
            raise
        finally:
            # Always attempt graceful shutdown
            await self._shutdown_components()

async def main():
    """Main entry point for the MCP server."""
    try:
        app = MCPServerApplication()
        await app.run()
        return 0
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        return 0
    except Exception as e:
        logger.error(f"Server failed: {e}")
        return 1

if __name__ == "__main__":
    # Run the server
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
