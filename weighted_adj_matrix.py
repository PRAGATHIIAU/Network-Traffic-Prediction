import pandas as pd
import numpy as np

def ip_to_binary(ip):
    """Convert an IP address to a 32-bit binary string."""
    return ''.join(f'{int(part):08b}' for part in ip.split('.'))

def common_prefix_length(ip1, ip2):
    """Calculate the length of the common prefix between two binary IPs."""
    ip1_bin = ip_to_binary(ip1)
    ip2_bin = ip_to_binary(ip2)
    
    # Calculate the length of the common prefix
    common_length = 0
    for b1, b2 in zip(ip1_bin, ip2_bin):
        if b1 == b2:
            common_length += 1
        else:
            break
    return common_length

def subnet_distance(ip1, ip2):
    """Calculate the subnet distance between two IP addresses."""
    max_length = 32  # IPv4 addresses have 32 bits
    common_length = common_prefix_length(ip1, ip2)
    return max_length - common_length

def create_weighted_adjacency_matrix(ips):
    """Create a weighted adjacency matrix using subnet distance for a list of IP addresses."""
    n = len(ips)
    weighted_adj_matrix = np.zeros((n, n))

    # Calculate subnet distances and fill the matrix
    for i in range(n):
        for j in range(n):
            if i != j:  # No self-loops
                weighted_adj_matrix[i][j] = subnet_distance(ips[i], ips[j])
    
    return weighted_adj_matrix

# Step 1: Read IP addresses from the CSV file
csv_file_path = 'selected_ips.csv'  # Replace with the path to your CSV file
df = pd.read_csv(csv_file_path)

# Ensure 'selectedips' column exists
if 'selectedips' not in df.columns:
    raise ValueError("The CSV file must have a 'selectedips' column.")

# Retrieve IP addresses from the 'selectedips' column
ips = df['selectedips'].dropna().tolist()

# Step 2: Create the weighted adjacency matrix
weighted_adj_matrix = create_weighted_adjacency_matrix(ips)

# Convert the matrix to a pandas DataFrame for saving
weighted_adj_df = pd.DataFrame(weighted_adj_matrix, index=ips, columns=ips)

# Step 3: Save the weighted adjacency matrix to a CSV file
weighted_adj_df.to_csv('weighted_adjacency_matrix.csv')
print("Weighted adjacency matrix saved as 'weighted_adjacency_matrix.csv'.")

# Step 4: Save the weighted adjacency matrix to a PKL (Pickle) file
weighted_adj_df.to_pickle('weighted_adjacency_matrix.pkl')
print("Weighted adjacency matrix saved as 'weighted_adjacency_matrix.pkl'.")
