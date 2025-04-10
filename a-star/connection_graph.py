import csv
import heapq
import math
import time
from datetime import datetime, timedelta


class Node:
    def __init__(self, name, lat, lon):
        self.name = name
        self.lat = lat
        self.lon = lon
        self.connections = []

    def add_connection(self, connection):
        self.connections.append(connection)

    def __repr__(self):
        return str(self.name) + ', ' + str(self.lat) + ', ' + str(self)


class Connection:
    def __init__(self, start_node: Node, end_node: Node, departure_time, arrival_time, line):
        self.start_node = start_node
        self.end_node = end_node
        self.departure_time = parse_time(departure_time)
        self.arrival_time = parse_time(arrival_time)
        self.minutes_time_diff = time_difference_minutes(departure_time, arrival_time)
        self.meters_diff = haversine_distance(start_node.lat, start_node.lon, end_node.lat, end_node.lon)
        self.line = line

    def __repr__(self):
        return str(self.start_node) + ', ' + str(self.end_node)

    def __lt__(self, other):
        return self.line == other.line


def time_difference_minutes(time1, time2):
    if not isinstance(time1, datetime):
        time1 = parse_time(time1)
    if not isinstance(time2, datetime):
        time2 = parse_time(time2)
    return (time2 - time1).total_seconds() / 60


def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    diff_lon = abs(lon2 - lon1)
    diff_lat = abs(lat2 - lat1)
    a = math.sin(diff_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(diff_lon / 2) ** 2
    distance_meters = 2 * math.asin(math.sqrt(a)) * 6371000

    return distance_meters


def parse_time(time_str):
    hour = int(time_str[:2])
    if hour >= 24:
        return datetime.strptime(str(hour - 24) + time_str[2:], "%H:%M:%S") + timedelta(days=1)
    else:
        return datetime.strptime(time_str, "%H:%M:%S")


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


def dijkstra(graph, start_name: str, end_name: str, start_time: datetime):
    start_time_calc = time.time()

    if start_name not in graph or end_name not in graph:
        print("Start node or end node does not exist")
        return None, 0, 0

    start_node = graph[start_name]
    end_node = graph[end_name]

    pq = [(0, None, start_time, None, start_node)]
    visited = set()
    time_costs = {start_name: 0}
    parent = {}

    while pq:
        current_time_cost, current_line, current_time, prev_connection, current_node = heapq.heappop(pq)

        if current_node in visited:
            continue

        if current_node == end_node:
            path = [(end_node.name, None, None, None)]
            node_name = current_node.name

            while node_name in parent:
                path.append(parent[node_name])
                node_name = parent[node_name][0]

            path.reverse()

            calculation_time = time.time() - start_time_calc
            return path, current_time_cost, calculation_time

        visited.add(current_node.name)

        for connection in current_node.connections:
            next_node = connection.end_node

            if connection.departure_time >= current_time:
                wait_time = connection.departure_time - current_time
                new_cost = int(wait_time.total_seconds() // 60) + connection.minutes_time_diff + current_time_cost

                if next_node.name not in time_costs or new_cost < time_costs[next_node.name]:
                    time_costs[next_node.name] = new_cost
                    parent[next_node.name] = (
                        current_node.name, connection.line, connection.departure_time, connection.arrival_time
                    )

                    heapq.heappush(pq, (new_cost, connection.line, connection.arrival_time, connection, next_node))

    print(f"No path from {start_node} to {end_node}")
    calculation_time = time.time() - start_time_calc
    return None, 0, calculation_time


def a_star(graph, start_name: str, end_name: str, start_time: datetime):
    start_time_calc = time.time()

    if start_name not in graph or end_name not in graph:
        print("Start node or end node does not exist")
        return None, 0, 0

    start_node = graph[start_name]
    end_node = graph[end_name]

    def heuristic(start_node, end_node):
        return haversine_distance(start_node.lat, start_node.lon, end_node.lat, end_node.lon) / 1000 * 0.25

    pq = [(0 + heuristic(start_node, end_node), 0, 0, None, start_time, None, start_node)]
    visited = set()
    time_costs = {start_name: 0}
    parent = {}

    while pq:
        heuristic_cost, current_time_cost, transfers, current_line, current_time, prev_connection, current_node = heapq.heappop(
            pq)

        if current_node in visited:
            continue

        if current_node == end_node:
            path = [(end_node.name, None, None, None)]
            node_name = current_node.name

            while node_name in parent:
                path.append(parent[node_name])
                node_name = parent[node_name][0]

            path.reverse()

            calculation_time = time.time() - start_time_calc
            return path, current_time_cost, calculation_time

        visited.add(current_node.name)

        for connection in current_node.connections:
            next_node = connection.end_node

            if connection.departure_time >= current_time:
                wait_time = connection.departure_time - current_time
                new_cost = int(wait_time.total_seconds() // 60) + time_difference_minutes(connection.departure_time, connection.arrival_time) + current_time_cost

                if next_node.name not in time_costs or new_cost < time_costs[next_node.name]:
                    time_costs[next_node.name] = new_cost
                    new_heuristic_cost = new_cost + heuristic(next_node, end_node)
                    parent[next_node.name] = (
                        current_node.name, connection.line, connection.departure_time, connection.arrival_time
                    )

                    new_transfers = transfers
                    if current_line is not None and connection.line != current_line:
                        new_transfers = transfers + 1

                    heapq.heappush(pq, (
                        new_heuristic_cost, new_cost, new_transfers, connection.line, connection.arrival_time, connection,
                        next_node))

    print(f"No path from {start_node} to {end_node}")
    calculation_time = time.time() - start_time_calc
    return None, 0, calculation_time


def a_star_improved(graph, start_name: str, end_name: str, start_time: datetime):
    amplifier = float(input("Enter the heuristic amplifier: "))
    start_time_calc = time.time()

    if start_name not in graph or end_name not in graph:
        print("Start node or end node does not exist")
        return None, 0, 0

    start_node = graph[start_name]
    end_node = graph[end_name]

    def heuristic(start_node, end_node):
        return haversine_distance(start_node.lat, start_node.lon, end_node.lat, end_node.lon) / 1000 * amplifier

    pq = [(0 + heuristic(start_node, end_node), 0, 0, None, start_time, None, start_node)]
    visited = set()
    time_costs = {start_name: 0}
    parent = {}

    while pq:
        heuristic_cost, current_time_cost, transfers, current_line, current_time, prev_connection, current_node = heapq.heappop(
            pq)

        if current_node in visited:
            continue

        if current_node == end_node:
            path = [(end_node.name, None, None, None)]
            node_name = current_node.name

            while node_name in parent:
                path.append(parent[node_name])
                node_name = parent[node_name][0]

            path.reverse()

            calculation_time = time.time() - start_time_calc
            return path, current_time_cost, calculation_time

        visited.add(current_node.name)

        for connection in current_node.connections:
            next_node = connection.end_node

            if connection.departure_time >= current_time:
                wait_time = connection.departure_time - current_time
                new_cost = int(wait_time.total_seconds() / 60) + connection.minutes_time_diff + current_time_cost
                new_heuristic_cost = new_cost + heuristic(next_node, end_node)

                new_transfers = transfers
                if current_line is not None or connection.line != current_line:
                    new_transfers += 1
                    new_cost += 1

                if next_node.name not in time_costs or new_cost < time_costs[next_node.name]:
                    time_costs[next_node.name] = new_cost

                    heapq.heappush(pq, (
                        new_heuristic_cost, new_cost, new_transfers, connection.line, connection.arrival_time, connection,
                        next_node))

                    parent[next_node.name] = (
                        current_node.name, connection.line, connection.departure_time, connection.arrival_time,
                        heuristic_cost, current_time_cost, transfers
                    )

    print(f"No path from {start_node} to {end_node}")
    calculation_time = time.time() - start_time_calc
    return None, 0, calculation_time


def a_star_transfers(graph, start_name: str, end_name: str, start_time: datetime):
    amplifier = float(input("Enter the heuristic amplifier: "))
    start_time_calc = time.time()

    if start_name not in graph or end_name not in graph:
        print("Start node or end node does not exist")
        return None, 0, 0

    start_node = graph[start_name]
    end_node = graph[end_name]

    def heuristic(start_node, end_node):
        return haversine_distance(start_node.lat, start_node.lon, end_node.lat, end_node.lon) / 1000 * amplifier

    pq = [(0, 0 + heuristic(start_node, end_node), 0, None, start_time, None, start_node)]
    visited = set()
    time_costs = {start_name: 0}
    transfer_costs = {start_name: 0}
    parent = {}

    while pq:
        transfers, heuristic_cost, current_time_cost, current_line, current_time, prev_connection, current_node = heapq.heappop(pq)

        if current_node in visited:
            continue

        if current_node == end_node:
            path = [(end_node.name, None, None, None)]
            node_name = current_node.name

            while node_name in parent:
                path.append(parent[node_name])
                node_name = parent[node_name][0]

            path.reverse()

            calculation_time = time.time() - start_time_calc
            return path, current_time_cost, calculation_time

        visited.add(current_node.name)

        for connection in current_node.connections:
            next_node = connection.end_node

            if connection.departure_time >= current_time:
                wait_time = connection.departure_time - current_time
                new_cost = int(wait_time.total_seconds() // 60) + connection.minutes_time_diff + current_time_cost

                new_transfers = transfers
                if current_line is not None and connection.line != current_line:
                    new_transfers = transfers + 1

                if (next_node.name not in transfer_costs or new_transfers < transfer_costs[next_node.name] or
                        (new_transfers == transfer_costs[next_node.name] and new_cost < time_costs[next_node.name])):
                    time_costs[next_node.name] = new_cost
                    transfer_costs[next_node.name] = new_transfers

                    new_heuristic_cost = new_cost + heuristic(next_node, end_node)

                    parent[next_node.name] = (
                        current_node.name, connection.line, connection.departure_time, connection.arrival_time
                    )

                    heapq.heappush(pq, (
                        new_transfers, new_heuristic_cost, new_cost, connection.line, connection.arrival_time, connection,
                        next_node))

    print(f"No path from {start_node} to {end_node}")
    calculation_time = time.time() - start_time_calc
    return None, 0, calculation_time


def print_stats(path, start_name, cost, calc_time):
    if path is None:
        return

    print("Starting from " + start_name)

    print("  Line\t | Departure\t | Arrival\t | Stops")
    for current_node, next_node in zip(path, path[1:]):
        print(f"\t{current_node[1]}\t"
              f" | {datetime.strftime(current_node[2], "%H:%M:%S")}\t\t"
              f" | {datetime.strftime(current_node[3], "%H:%M:%S")}\t"
              f" | {current_node[0]} -> {next_node[0]}")
        # print(f"{current_node[6]}\t{current_node[4]}\t{current_node[5]}")

        if current_node[1] != next_node[1] and next_node[1] is not None:
            print("Warning! Line change")

    print(f"Total time cost: \t\t\t{cost} minutes")
    print(f"Total calculation time: \t{calc_time * 1000} ms")


def main():
    start_time_calc = time.time()
    graph = load_and_build_graph()
    print(f"Graph loaded in {time.time() - start_time_calc} seconds")

    print("\n--- Jakdojade ---\n")
    while True:
        task = input("Enter task to start 1/2 (or 'q' to quit): ")

        if task == 'q':
            print("Goodbye!")
            break

        if task == '1':
            start_node = input("Enter start node: ")
            end_node = input("Enter end node: ")
            start_time = input("Enter start time HH:MM:SS: ")
            optimization_criterion = input("Enter optimization p/t: ")

            if optimization_criterion == 't':
                path, cost, calc_time = a_star_improved(graph, start_node, end_node, parse_time(start_time))
                _, d_cost, d_calc_time = dijkstra(graph, start_node, end_node, parse_time(start_time))
                _, a_cost, a_calc_time = a_star(graph, start_node, end_node, parse_time(start_time))
                print_stats(path, start_node, cost, calc_time)

                print("\n--- Comparison with usual A-star ---")
                print(f"A-star time cost: \t\t\t{a_cost} minutes")
                print(f"A-star calculation time: \t{a_calc_time * 1000} ms")

                print("--- Comparison with Dijkstra ---")
                print(f"Dijkstra time cost: \t\t{d_cost} minutes")
                print(f"Dijkstra calculation time: \t{d_calc_time * 1000} ms\n")

            elif optimization_criterion == 'p':
                path, cost, calc_time = a_star_transfers(graph, start_node, end_node, parse_time(start_time))
                print_stats(path, start_node, cost, calc_time)
            else:
                print("Wrong optimization criteria")


if __name__ == "__main__":
    main()
"""
GRABISZYŃSKA (Cmentarz)
Krakowska
15:05:00

Psie Pole (Rondo Lotników Polskich)
FAT
12:55:00
"""
