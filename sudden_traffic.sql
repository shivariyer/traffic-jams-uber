WITH A AS
     (SELECT s.*,
             lead(phase) OVER
             (partition by osm_way_id order by utc_timestamp) lead_phase,
             lag(utc_timestamp) OVER
             (partition by osm_way_id order by utc_timestamp) lag_timestamp
      FROM speed s),
   B AS (SELECT s2.osm_way_id,
                s2.utc_timestamp,
                MIN(s0.utc_timestamp),
                EXTRACT(EPOCH FROM MIN(s0.utc_timestamp) - s2.utc_timestamp)/3600 cons
         FROM speed s2
              JOIN A s0  ON (s2.osm_way_id = s0.osm_way_id
                             AND s0.phase = 0
                             AND s0.utc_timestamp > s2.utc_timestamp
                             and s0.lead_phase <> 0)
         WHERE s2.sudden_traffic
         GROUP BY s2.osm_way_id,s2.utc_timestamp)
     UPDATE speed s3
     SET consecutive_hours = b.cons
     FROM B b
     WHERE s3.osm_way_id = b.osm_way_id
     AND s3.utc_timestamp = b.utc_timestamp;
