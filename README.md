```
sinfo -N -l
```


```
sbatch --nodelist=nodegpu031 job_script.sh
```




m progress

```
squeue -u $USER  # Check running jobs
tail -f output.txt  # View live output
tail -f error.txt  # View errors
```