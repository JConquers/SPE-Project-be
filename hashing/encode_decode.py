import hashlib
import base64
import re
import json
from typing import Dict, Any, Optional

class AadhaarDataEncoder:
    def __init__(self):
        self.data_store = {}

    def validate_aadhaar(self, aadhaar_number: str) -> bool:
        """Validate Aadhaar number format"""
        return bool(re.match(r'^\d{12}$', aadhaar_number))

    def encode_data(self, aadhaar_number: str, data_points: Dict[str, Any]) -> str:
        """
        Encode data specifically tied to Aadhaar
        
        Args:
            aadhaar_number: 12-digit Aadhaar number
            data_points: Dictionary of data to be encoded
        
        Returns:
            Unique hash identifier
        """
        # Validate Aadhaar
        if not self.validate_aadhaar(aadhaar_number):
            raise ValueError("Invalid Aadhaar number")

        # Generate unique hash using Aadhaar as primary key
        unique_hash = hashlib.sha3_256(
            (aadhaar_number + json.dumps(data_points, sort_keys=True)).encode()
        ).hexdigest()
        
        # Store data with Aadhaar-based hash
        self.data_store[unique_hash] = {
            'aadhaar': aadhaar_number,
            'data': data_points
        }
        
        return unique_hash

    def decode_data(self, unique_hash: str, input_aadhaar: str) -> Optional[Dict[str, Any]]:
        """
        Decode data by verifying Aadhaar
        
        Args:
            unique_hash: Generated unique hash
            input_aadhaar: Aadhaar number for verification
        
        Returns:
            Decoded data if verification successful
        """
        # Validate input Aadhaar
        if not self.validate_aadhaar(input_aadhaar):
            raise ValueError("Invalid Aadhaar number")

        # Check hash exists and Aadhaar matches
        if (unique_hash not in self.data_store or 
            self.data_store[unique_hash]['aadhaar'] != input_aadhaar):
            return None
        
        return self.data_store[unique_hash]['data']

def main():
    # Initialize encoder
    encoder = AadhaarDataEncoder()

    # Sample data points
    sample_data = {
        'community_reputation': 'Good',
        'business_activity': 50,
        'credit_relationships': 'Good credit history',
        'asset_ownership': ['sewing machine', 'bicycle']
    }

    # Aadhaar number for encoding
    aadhaar_number = '123456789012'

    try:
        # Encode data
        unique_hash = encoder.encode_data(aadhaar_number, sample_data)
        print(f"Generated Hash: {unique_hash}")

        # Decode data with correct Aadhaar
        retrieved_data = encoder.decode_data(unique_hash, aadhaar_number)
        print("Retrieved Data:", retrieved_data)

        # Attempt decode with incorrect Aadhaar
        incorrect_retrieve = encoder.decode_data(unique_hash, '000000000000')
        print("Incorrect Decode:", incorrect_retrieve)

    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()