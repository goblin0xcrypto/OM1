import logging
import requests
from typing import Optional
from pydantic import Field
from tenacity import retry, stop_after_attempt, wait_exponential
from actions.base import ActionConfig, ActionConnector
from actions.gps.interface import GPSAction, GPSInput
from providers.io_provider import IOProvider

class GPSFabricConfig(ActionConfig):
    """
    Configuration for GPS Fabric connector.
    
    Parameters
    ----------
    fabric_endpoint : str
        The endpoint URL for the Fabric network.
    request_timeout : int
        Timeout for HTTP requests in seconds.
    max_retries : int
        Maximum number of retry attempts.
    """
    fabric_endpoint: str = Field(
        default="http://localhost:8545",
        description="The endpoint URL for the Fabric network.",
    )
    request_timeout: int = Field(
        default=10,
        description="Timeout for HTTP requests in seconds.",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts.",
    )

class GPSFabricConnector(ActionConnector[GPSFabricConfig, GPSInput]):
    """
    Connector that shares GPS coordinates via a Fabric network.
    """
    
    def __init__(self, config: GPSFabricConfig):
        """
        Initialize the GPSFabricConnector.
        
        Parameters
        ----------
        config : GPSFabricConfig
            Configuration for the action connector.
        """
        super().__init__(config)
        self.io_provider = IOProvider()
        self.fabric_endpoint = self.config.fabric_endpoint
        self.request_timeout = self.config.request_timeout
        self.max_retries = self.config.max_retries
    
    async def connect(self, output_interface: GPSInput) -> None:
        """
        Connect to the Fabric network and send GPS coordinates.
        
        Parameters
        ----------
        output_interface : GPSInput
            The GPS input containing the action to be performed.
        """
        logging.info(f"GPSFabricConnector: Action requested - {output_interface.action}")
        
        if output_interface.action == GPSAction.SHARE_LOCATION:
            success = self.send_coordinates()
            if not success:
                logging.warning("GPSFabricConnector: Failed to share location")
    
    def _get_coordinates(self) -> Optional[dict]:
        """
        Retrieve GPS coordinates from IO provider.
        
        Returns
        -------
        Optional[dict]
            Dictionary with latitude, longitude, and yaw, or None if incomplete.
        """
        latitude = self.io_provider.get_dynamic_variable("latitude")
        longitude = self.io_provider.get_dynamic_variable("longitude")
        yaw = self.io_provider.get_dynamic_variable("yaw_deg")
        
        logging.debug(f"GPSFabricConnector: Retrieved coordinates - "
                     f"lat: {latitude}, lon: {longitude}, yaw: {yaw}")
        
        # Check if ANY coordinate is missing
        if latitude is None or longitude is None or yaw is None:
            logging.error(f"GPSFabricConnector: Incomplete coordinates - "
                         f"lat: {latitude}, lon: {longitude}, yaw: {yaw}")
            return None
        
        return {
            "latitude": latitude,
            "longitude": longitude,
            "yaw": yaw
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def _send_request(self, coordinates: dict) -> requests.Response:
        """
        Send HTTP request to Fabric network with retry logic.
        
        Parameters
        ----------
        coordinates : dict
            GPS coordinates to send.
            
        Returns
        -------
        requests.Response
            HTTP response object.
        """
        return requests.post(
            self.fabric_endpoint,
            json={
                "method": "omp2p_shareStatus",
                "params": [coordinates],
                "id": 1,
                "jsonrpc": "2.0",
            },
            headers={"Content-Type": "application/json"},
            timeout=self.request_timeout,
        )
    
    def send_coordinates(self) -> bool:
        """
        Send GPS coordinates to the Fabric network.
        
        Returns
        -------
        bool
            True if coordinates were sent successfully, False otherwise.
        """
        logging.info("GPSFabricConnector: Preparing to send coordinates to Fabric network")
        
        # Get coordinates
        coordinates = self._get_coordinates()
        if coordinates is None:
            return False
        
        try:
            # Send request with retry logic
            response = self._send_request(coordinates)
            
            # Check HTTP status
            response.raise_for_status()
            
            # Parse JSON response
            try:
                result = response.json()
            except ValueError as e:
                logging.error(f"GPSFabricConnector: Invalid JSON response: {e}")
                logging.debug(f"Response content: {response.text}")
                return False
            
            # Check JSON-RPC result
            if "result" in result and result["result"]:
                logging.info("GPSFabricConnector: Coordinates shared successfully")
                return True
            elif "error" in result:
                logging.error(f"GPSFabricConnector: JSON-RPC error: {result['error']}")
                return False
            else:
                logging.error(f"GPSFabricConnector: Unexpected response format: {result}")
                return False
                
        except requests.Timeout:
            logging.error(f"GPSFabricConnector: Request timeout after {self.request_timeout}s")
            return False
        except requests.ConnectionError as e:
            logging.error(f"GPSFabricConnector: Connection error: {e}")
            return False
        except requests.HTTPError as e:
            logging.error(f"GPSFabricConnector: HTTP error {e.response.status_code}: {e}")
            return False
        except requests.RequestException as e:
            logging.error(f"GPSFabricConnector: Request error: {e}")
            return False
        except Exception as e:
            logging.error(f"GPSFabricConnector: Unexpected error: {e}", exc_info=True)
            return False