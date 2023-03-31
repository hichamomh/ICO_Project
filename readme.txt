
This dataset contains 9 Excel (xls) files that can be used as a new benchmark data for the solving of real-world vehicle routing problems with realistic non-standard constraints. All data are real and obtained experimentally by using VRP algorithm on production environment in one of the biggest distribution company in Bosnia and Herzegovina.

The files with description are:

1_master_table_route_settings.xls - contains several basic setting information about the 11 created real-world VRP routes. Each route is uniquely identified by field ROUTE_ID, and all other detail tables have that field. Column with prefix RESULT_ are set during execution of the algorithm.
2_detail_table_customers.xls - this file contains input information about customers who need to be serviced during the delivery of ordered items. Each customer for one route is uniquely identified with field CUSTOMER_CODE.
3_detail_table_vehicles.xls - contains all necessary information about available fleet of vehicles. Each vehicle for one route is uniquely identified with field VEHICLE_CODE. Columns with prefix RESULT_ are set during execution of the algorithm.
4_detail_table_depots.xls - contains all necessary information about available depots for each routing. Each depot for one route is uniquely identified with field DEPOT_CODE. 
5_detail_table_constraints_sdvrp.xls - in this file there are all constraints for each od 11 routing which customer (identified by CUSTOMER_CODE) could not be serviced by which vehicles (identified by VEHICLE_CODE) - SDVRP constraints.
6_detail_table_cust_depots_distances.xls - this file contains real distances (travel and time distances) between depots (identified by DEPOT_CODE) and each of the customers (identified by CUSTOMER_CODE) for all routings, and reverse between each customer and depots. If in the DIRECTION column is value DEPOT->CUSTOMER then the distances are from DEPOT to the CUSTOMER, and in the other way (DIRECTION = CUSTOMER->DEPOT) then the distances are between CUSTOMER and DEPOT. In real environment that distances are not the same. All distances in this file are obtained using GraphHopper API, and OpenStreetMap with included constraints in 9_table_blocked_parts_of_the_road.xls.
7_detail_table_cust_cust_distances.xls - this file contains real distances (travel and time distances) between each of the customers (identified by CUSTOMER_CODE_FROM and CUSTOMER_CODE_TO) mutually for all routings. All distances in this file are obtained using GraphHopper API, and OpenStreetMap with included constraints in 9_table_blocked_parts_of_the_road.xls.. File contains 3 sheets, because of its length.
8_detail_table_routes_RESULTS.xls - this file contains obtained results for each of the 11 routes, by using proposed "Innovative modular approach for solving real-world vehicle routing problems with realistic non-standard constraints".

9_table_blocked_parts_of_the_road.xls - this file is using during the real calculations of distances (travel and time), and contains the blocking parts of the road for all vehicles which must be excluded - because of that roads are not in good state and it is not recommended that the delivery be done through these roads.

Units of measure:
KM - Convertible mark (currency in Bosnia and Herzegovina), and in the case of Distances KM means Kilometers
MIN - Minutes
KG - Kilograms
M3 - Cubic meters
