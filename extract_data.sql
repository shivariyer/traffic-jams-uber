SELECT osm_way_id,
       FLOOR(EXTRACT(HOUR FROM utc_timestamp)) AS hour_num ,
       speed.AVG AS hour_speed
FROM speed
WHERE osm_way_id = 4962952
