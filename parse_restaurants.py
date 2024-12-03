import json

def filter_restaurants_by_location(file_path, location, output_file):
    try:
        # Open and load the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Ensure the data is a list
        if not isinstance(data, list):
            raise ValueError("JSON data should be a list of objects.")
        
        # Filter restaurants with the specified location
        san_diego_restaurants = [
            restaurant for restaurant in data 
            if restaurant.get('location') == location
        ]
        
        # Write the filtered results to a new JSON file
        with open(output_file, 'w') as out_file:
            json.dump(san_diego_restaurants, out_file, indent=4)
        
        print(f"Filtered results have been saved to {output_file}")

    except FileNotFoundError:
        print("The specified file does not exist.")
    except json.JSONDecodeError:
        print("Invalid JSON format.")
    except Exception as e:
        print(f"An error occurred: {e}")

input_file = 'restaurants.json'  # Replace with file name
output_file = 'san_diego_restaurants.json'  # Name of the output file
location = "San Diego"
filter_restaurants_by_location(input_file, location, output_file)
