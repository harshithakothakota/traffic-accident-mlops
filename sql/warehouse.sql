-- Star schema + analytical views, rebuilt from the cleaned tables
-- (collisions, vehicles, casualties) on every pipeline run.

DROP TABLE IF EXISTS fact_accidents, dim_date, dim_location, dim_road,
                     dim_conditions, dim_severity CASCADE;

-- ---------------------------------------------------------- dimensions
CREATE TABLE dim_date AS
SELECT DISTINCT
    datetime::timestamp::date                        AS date,
    extract(year  FROM datetime::timestamp)::int     AS year,
    extract(month FROM datetime::timestamp)::int     AS month,
    trim(to_char(datetime::timestamp, 'Month'))      AS month_name,
    trim(to_char(datetime::timestamp, 'Day'))        AS weekday,
    extract(isodow FROM datetime::timestamp)::int    AS weekday_number,   -- 1 = Monday
    extract(isodow FROM datetime::timestamp) >= 6    AS is_weekend
FROM collisions;
ALTER TABLE dim_date ADD PRIMARY KEY (date);

CREATE TABLE dim_severity (severity INT PRIMARY KEY, severity_label TEXT);
INSERT INTO dim_severity VALUES (1, 'Fatal'), (2, 'Serious'), (3, 'Slight');

CREATE TABLE dim_location AS
SELECT row_number() OVER () AS location_id, *
FROM (SELECT DISTINCT police_force, police_force_name, district, urban_or_rural_area,
             CASE urban_or_rural_area WHEN 1 THEN 'Urban' WHEN 2 THEN 'Rural'
                  ELSE 'Unknown' END AS area_type
      FROM collisions) x;
ALTER TABLE dim_location ADD PRIMARY KEY (location_id);

CREATE TABLE dim_road AS
SELECT row_number() OVER () AS road_id, *
FROM (SELECT DISTINCT road_type, speed_limit,
             CASE road_type WHEN 1 THEN 'Roundabout' WHEN 2 THEN 'One way street'
                  WHEN 3 THEN 'Dual carriageway' WHEN 6 THEN 'Single carriageway'
                  WHEN 7 THEN 'Slip road' ELSE 'Unknown' END AS road_type_label
      FROM collisions) x;
ALTER TABLE dim_road ADD PRIMARY KEY (road_id);

CREATE TABLE dim_conditions AS
SELECT row_number() OVER () AS conditions_id, *
FROM (SELECT DISTINCT weather_conditions, road_surface_conditions, light_conditions,
             CASE weather_conditions WHEN 1 THEN 'Fine' WHEN 2 THEN 'Raining'
                  WHEN 3 THEN 'Snowing' WHEN 4 THEN 'Fine + high winds'
                  WHEN 5 THEN 'Raining + high winds' WHEN 6 THEN 'Snowing + high winds'
                  WHEN 7 THEN 'Fog or mist' WHEN 8 THEN 'Other' ELSE 'Unknown' END AS weather_label,
             CASE road_surface_conditions WHEN 1 THEN 'Dry' WHEN 2 THEN 'Wet or damp'
                  WHEN 3 THEN 'Snow' WHEN 4 THEN 'Frost or ice' WHEN 5 THEN 'Flood'
                  ELSE 'Unknown' END AS surface_label,
             CASE WHEN light_conditions = 1 THEN 'Daylight'
                  WHEN light_conditions IN (4, 5, 6, 7) THEN 'Darkness'
                  ELSE 'Unknown' END AS light_label
      FROM collisions) x;
ALTER TABLE dim_conditions ADD PRIMARY KEY (conditions_id);

-- ---------------------------------------------------------------- fact
CREATE TABLE fact_accidents AS
SELECT c.collision_index,
       c.datetime::timestamp::date                 AS date,
       extract(hour FROM c.datetime::timestamp)::int AS hour,
       c.severity,
       l.location_id, r.road_id, k.conditions_id,
       c.number_of_vehicles, c.number_of_casualties,
       c.temperature_c, c.precipitation_mm, c.wind_speed_kmh,
       c.latitude, c.longitude
FROM collisions c
JOIN dim_location l ON l.police_force IS NOT DISTINCT FROM c.police_force
                   AND l.district IS NOT DISTINCT FROM c.district
                   AND l.urban_or_rural_area IS NOT DISTINCT FROM c.urban_or_rural_area
JOIN dim_road r ON r.road_type IS NOT DISTINCT FROM c.road_type
             AND r.speed_limit IS NOT DISTINCT FROM c.speed_limit
JOIN dim_conditions k ON k.weather_conditions IS NOT DISTINCT FROM c.weather_conditions
                     AND k.road_surface_conditions IS NOT DISTINCT FROM c.road_surface_conditions
                     AND k.light_conditions IS NOT DISTINCT FROM c.light_conditions;
ALTER TABLE fact_accidents ADD PRIMARY KEY (collision_index);
ALTER TABLE fact_accidents
    ADD FOREIGN KEY (date) REFERENCES dim_date,
    ADD FOREIGN KEY (severity) REFERENCES dim_severity,
    ADD FOREIGN KEY (location_id) REFERENCES dim_location,
    ADD FOREIGN KEY (road_id) REFERENCES dim_road,
    ADD FOREIGN KEY (conditions_id) REFERENCES dim_conditions;

-- ------------------------------------------------- analytical mart (views)
-- one wide, easy-to-query view that the dashboard filters on
CREATE VIEW mart_accidents AS
SELECT f.*, d.month, d.month_name, d.weekday, d.weekday_number, d.is_weekend,
       s.severity_label, l.police_force_name, l.district, l.area_type, r.road_type_label, r.speed_limit,
       k.weather_label, k.surface_label, k.light_label
FROM fact_accidents f
JOIN dim_date d USING (date)
JOIN dim_severity s USING (severity)
JOIN dim_location l USING (location_id)
JOIN dim_road r USING (road_id)
JOIN dim_conditions k USING (conditions_id);

CREATE VIEW mart_by_hour AS
SELECT hour, severity_label, count(*) AS accidents
FROM mart_accidents GROUP BY 1, 2;

CREATE VIEW mart_by_weekday AS
SELECT weekday, weekday_number, count(*) AS accidents
FROM mart_accidents GROUP BY 1, 2;

CREATE VIEW mart_by_weather AS
SELECT weather_label, count(*) AS accidents,
       round(100.0 * avg((severity <= 2)::int), 1) AS pct_fatal_or_serious
FROM mart_accidents GROUP BY 1;

CREATE VIEW mart_by_road_type AS
SELECT road_type_label, count(*) AS accidents,
       round(100.0 * avg((severity <= 2)::int), 1) AS pct_fatal_or_serious
FROM mart_accidents GROUP BY 1;

CREATE VIEW mart_top_areas AS
SELECT police_force_name AS area, count(*) AS accidents,
       sum(number_of_casualties) AS casualties,
       count(*) FILTER (WHERE severity = 1) AS fatal
FROM mart_accidents GROUP BY 1 ORDER BY accidents DESC;
