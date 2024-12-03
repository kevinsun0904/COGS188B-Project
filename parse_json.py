def fix_invalid_json(input_file, output_file):
    try:
        # Read the invalid JSON file
        with open(input_file, 'r') as file:
            lines = file.readlines()

        # Add opening and closing brackets, and insert commas between objects
        fixed_json = []
        fixed_json.append("[")  # Start of JSON array
        
        for line in lines:
            stripped_line = line.strip()
            if stripped_line:  # Skip empty lines
                if stripped_line.endswith("}"):
                    fixed_json.append(stripped_line + ",")  # Add a comma at the end
        
        # Remove the last comma and add closing bracket
        if fixed_json[-1].endswith(","):
            fixed_json[-1] = fixed_json[-1][:-1]  # Remove trailing comma
        fixed_json.append("]")  # End of JSON array

        # Write the fixed JSON to the output file
        with open(output_file, 'w') as file:
            file.write("\n".join(fixed_json))

        print(f"Fixed JSON has been saved to {output_file}")
    
    except FileNotFoundError:
        print("The specified file does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
input_file = 'invalid_restaurants.json'  # Replace with your input file
output_file = 'fixed_restaurants.json'  # Name of the output file
fix_invalid_json(input_file, output_file)
