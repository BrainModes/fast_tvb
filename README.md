# fast_tvb
Fast and parallel C implementation of TVB

## Pull and run fast_tvb as a Docker container here
## https://hub.docker.com/r/thevirtualbrain/fast_tvb


The code implements a brain network model composed of connected ReducedWongWang neural mass models (Wong & Wang, 2006) with feedback inhibition control (FIC). For more information on the model and FIC please see Deco et al. (2014), Schirner et al. (2018) and Shen et al. (2019).

For more information on The Virtual Brain (TVB) please see 
www.thevirtualbrain.org

For questions and other enquiries please contact 

Michael Schirner (michael.schirner@charite.de) or 
Petra Ritter (petra.ritter@charite.de).

# Generating the Docker container

If you want to generate the Docker container from scratch, two steps are necessary: first, compiling the C file, and, second, generating a Docker container with the new binary.

## Step 1: Compile tvbii_multicore.c

We will use a Docker container to compile the C code, see folder `step1_compile_C_code`.
1. Make sure you have the Docker client installed and running (https://www.docker.com/products/docker-desktop)
2. Download/pull this repository onto your local machine.
3. Create a folder `/path/to/tvb_binary` where the compiled binary will be stored.
4. Open a command line shell and enter the directory `step1_compile_C_code`.
5. Run the following:
```
docker build -t tvb_c .
docker run --rm --mount type=bind,source=/path/to/tvb_binary,target=/output tvb_c /compile_and_copy.sh
cp /path/to/tvb_binary/tvb ../step2_create_Docker_container/
cat /path/to/tvb_binary/build_output.txt
```
The first command builds the Docker container with the build environment for compiling the C file. The second command compiles the C file and copies the output into the folder `/path/to/tvb_binary` in your local filesystem. The third command copies the created binary `tvb` into the folder `step2_create_Docker_container`. The fourth command lets you inspect the output of the compiler -- closely inspect whether it contains error messages or warning that need to be fixed.  

## Step 2: Create container for fast_tvb

6. Now that our fast_tvb binary is created, we will create a Docker container that contains the binary for convenient execution on different platforms. Enter the folder `step2_create_Docker_container`.  

7. To build the Docker file run
```
docker build -t <your repo>/fast_tvb .
```
replacing `<your repo>` with the name of your Dockerhub repository.  

8. To push it into your repository run
```
docker push <your repo>/fast_tvb
``` 

9. To run `fast_tvb` run
```
docker run --rm --mount type=bind,source=/path/to/step2_create_Docker_container/output,target=/output --mount type=bind,source=/path/to/step2_create_Docker_container/input,target=/input  <your repo>/fast_tvb /start_simulation.sh param_set.txt gavg 4
```
The folder `step2_create_Docker_container` contains a demo brain network model in the folder `input`.


For more information on how to use `fast_tvb`, please follow the instructions at https://hub.docker.com/r/thevirtualbrain/fast_tvb.

  
# References
  
  Deco, G., Ponce-Alvarez, A., Hagmann, P., Romani, G. L., Mantini, D., & Corbetta, M. (2014). How local excitation–inhibition ratio impacts the whole brain dynamics. Journal of Neuroscience, 34(23), 7886-7898.
  
  Schirner, M., McIntosh, A. R., Jirsa, V., Deco, G., & Ritter, P. (2018). Inferring multi-scale neural mechanisms with brain network modelling. Elife, 7, e28927.
  
  Schirner, M., Domide, L., Perdikis, D., Triebkorn, P., Stefanovski, L., Pai, R., ... & Ritter, P. (2022). Brain simulation as a cloud service: The Virtual Brain on EBRAINS. NeuroImage, 118973.
  
  Shen, K., Bezgin, G., Schirner, M., Ritter, P., Everling, S., McIntosh, A. R. (2019) A macaque connectome for large-scale network simulations in TheVirtualBrain. (under review)
  
  Wong, K. F., & Wang, X. J. (2006). A recurrent network mechanism of time integration in perceptual decisions. Journal of Neuroscience, 26(4), 1314-1328.
 

# Acknowledgments

This  research  has  received  funding  from  the  European  Union’s  Horizon  2020  Framework  Programme  for  Research  and  Innovation  under  the  Specific  Grant  Agreement  Nos.  785907  (Human  Brain Project SGA2),  945539  (Human  Brain Project SGA3), ICEI 800858, VirtualBrainCloud 826421 and ERC 683049.
