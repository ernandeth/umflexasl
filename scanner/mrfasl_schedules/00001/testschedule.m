deadtime = ones(100,1)*0.1;

pcasl_type = ones(100,1);
pcasl_type(1:3:end) = 0;
pcasl_type(2:3:end) = -1;

pcasl_duration = ones(100,1)*0.2;
pcasl_pld = ones(100,1)*0.2;

p1_type = ones(100,1);
p1_type(1:3:end) = 0;
p1_type(2:3:end) = -1;
p1_pld = ones(100,1)*0.2;


p2_type = ones(100,1);
p2_type(1:3:end) = 0;
p2_type(2:3:end) = -1;
p2_pld = ones(100,1)*0.2;

schedule = [deadtime pcasl_type pcasl_duration pcasl_pld p1_type p1_pld p2_type p2_pld]

save mrf_schedule.txt schedule -ascii