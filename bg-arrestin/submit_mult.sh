for SEED in $(seq 0 4); do 
    cp submit-template.sb submit-template${SEED}.sb
    sed -i "s/SEED/${SEED}/g" submit-template${SEED}.sb
    sbatch submit-template${SEED}.sb
done
