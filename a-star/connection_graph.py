import csv


class Node:
    def __init__(self, name, lat, lon):
        self.name = name
        self.lat = lat
        self.lon = lon
        self.connections = []

    def add_connection(self, connection):
        self.connections.append(connection)


class Connection:
    def __init__(self, start_node, end_node, departure_time, arrival_time, line):
        self.start_node = start_node
        self.end_node = end_node
        self.departure_time = departure_time
        self.arrival_time = arrival_time
        self.line = line


def load_and_build_graph():
    stop_name_to_node = {}

    with open('connection_graph.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            start_name = row['start_stop']
            end_name = row['end_stop']

            if start_name not in stop_name_to_node:
                stop_name_to_node[start_name] = Node(
                    start_name,
                    float(row['start_stop_lat']),
                    float(row['start_stop_lon'])
                )

            if end_name not in stop_name_to_node:
                stop_name_to_node[end_name] = Node(
                    end_name,
                    float(row['end_stop_lat']),
                    float(row['end_stop_lon'])
                )

            start_node = stop_name_to_node[start_name]
            end_node = stop_name_to_node[end_name]

            connection = Connection(
                start_node=start_node,
                end_node=end_node,
                departure_time=row['departure_time'],
                arrival_time=row['arrival_time'],
                line=row['line']
            )

            start_node.add_connection(connection)

    return stop_name_to_node
