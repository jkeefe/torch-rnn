# My process notes for this branch

## Spun up a "deep learning" AMI on Amazon EC2

- Went to AWS Dashboard
- Clicked "Launch Instance"
- Used the "AWS marketplace" and searched for "deep learning"
- Picked the "Deep Learning AMI Ubuntu Version"
- Spun it up into a `p2.xlarge` instance, which is about 90 cents an hour
- Tried to do as a spot instance, but couldn't because it was grayed-out. Sent Amazon a ticket asking about this
- ssh'd into the server

## Getting Torch running

- the `th` command didn't work out of the gate ... tho the AMI comes with Torch. 🤔
- went into `~/src/torch` and ran the `install-deps` as described in the [Torch install docs](http://torch.ch/docs/getting-started.html#_), but all dependencies were already installed.
- so in the same directory did `./install.sh` ... which clearly installed what I needed, and took advantage of the CUDA and GPUs in the box
- then exited the instance and re-logged in to get everything going
- now the `th` gets me:

```
 ______             __   |  Torch7 
/_  __/__  ________/ /   |  Scientific computing for Lua. 
 / / / _ \/ __/ __/ _ \  |  Type ? for help 
/_/  \___/_/  \__/_//_/  |  https://github.com/torch 
                         |  http://torch.ch 
  
th>
```

Yay!

NOTE: I checked the luarocks packages using `luarocks list` and all of the packages listed in the [torch-rnn repo](https://github.com/jcjohnson/torch-rnn) were already installed. Sweet. Same for `python2.7-dev` and `libhdf5-dev`.

The only thing missing was the `torch-hdf5`, which the [instructions](https://github.com/jcjohnson/torch-rnn) say we need form GitHub:

```
# We need to install torch-hdf5 from GitHub
git clone https://github.com/deepmind/torch-hdf5
cd torch-hdf5
luarocks make hdf5-0-0.rockspec
```

That worked.

## Get GitHub working

- made a new ssh key on the EC2 using this page: https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/
- added it as an authorized key in my settings in GitHub
- on the web, forked the torch-rnn repo to my own account
- then cloned [my fork](https://github.com/jkeefe/torch-rnn) onto the EC2 instance 
- created a new branch called dognames using `git branch dognames` and `git checkout dognames`
- worked in this branch, being sure to push with `-u` to push the branch upstream: `git push -u origin dognames`

## Getting conda environment working

- In retospect, this may not have been necessary. But did it anyway to maintain good practice.
- couldn't use `conda` until added this to `.bashrc`
    `export PATH="/home/ubuntu/src/anaconda2/bin:$PATH"`
- but `conda create --name dogs` ran into permissions problems
- but then couldnt use `sudo conda`. Sigh. 
- So created new env using full path to conda:
    ` sudo /home/ubuntu/src/anaconda2/bin/conda create --name dogs`
- Then: `source activate dogs`


## Prepped the data

- Downloaded the 2013 NYC dog dataset from [this WNYC fusion table](https://fusiontables.google.com/data?docid=1pKcxc8kzJbBVzLu_kgzoAMzqYhZyUhtScXjB0BQ#rows:id=1) which has 81,542 rows and 13,803 unique dog names. I didn't do anything to pull out just the unique names yet.
- used LibreOffice to take just the first column (the dog names) and put it into dogs.txt
- Now starting to follow [steps](https://github.com/jkeefe/torch-rnn#usage) outlined in the torch-rnn README.

```
python scripts/preprocess.py \
  --input_txt data/dogs.txt \
  --output_h5 data/dogs.h5 \
  --output_json data/dogs.json
```

Woot! Got this:

```
Total vocabulary size: 72
Total tokens in file: 507828
  Training size: 406264
  Val size: 50782
  Test size: 50782
Using dtype  <type 'numpy.uint8'>
```

## Training the model

Following along in the example instructions:

`th train.lua -input_h5 data/dogs.h5 -input_json data/dogs.json`

This took a bit of time, but that was fine. 

## Taking a sample!

`th sample.lua -checkpoint cv/checkpoint_8000.t7 -length 3000 > data/dogs_sample.txt`

Woot! Got cool new dog names! Though lots are clearly names that are in the database. 

## Filtering out existing names

To get a list of names made by the computer but not in the original data, I decided to whip up a quick python script in jupyter notebook.

```
conda install jupyter
pip install agate
jupyter notebook
```

I copy-pasted the URL I got into a browser to do the work.

The notebook is [here](https://github.com/jkeefe/torch-rnn/blob/dognames/compare_sample_to_original.ipynb).

# Russian bot tweets

## Prepping the data

Working from the "streamlined spreadsheet" data set here: https://www.nbcnews.com/tech/social-media/now-available-more-200-000-deleted-russian-troll-tweets-n844731

Don't install conda again! :-)

```
/Users/jkeefe/anaconda/bin/conda create --name rnn
source activate rnn
```

Then cutting down the csv:

Get the names of the columns: `csvcut -n tweets.csv`

Take just the `text` column name into a new csv: `csvcut -c text tweets.csv > justtext.csv`

Rename it, since it's just text now: `mv justtext.csv justtext.txt`

Using `sed` to clean up the text file, with guide to `sed` here: http://www.grymoire.com/Unix/Sed.html

- `-E` for regular expression searching
- `-e` for multiple expressions (one for the start of the line one for the end)
- `'s/^"//'` for quotes at the start of the line
- `'s/"$//'` quotes at the end
- `'s/""/"/g'` doublequotes to single ones (from the csv)
- `'s/…//g'` the ... character

```
sed -E -e 's/^"//' -e 's/"$//' -e 's/""/"/g' -e 's/…//g' <justtext.txt >cleaned_text.txt
```

Also eliminating links, just used this in atom: `[ ]?http[s]?\S+` (with an optional leading space).

Then eliminated blank lines in atom :`^\n`

Decided to also eliminate "RT @username: " ... so we really just get the TEXT that was propogated, no matter what the source. `^RT @\w+: `

## Deep Learning

Fired up the EC2 server!

So as not to mess up my dogs work ... and because I didn't manage to change branches ... once on the EC2 server I'm making a branch off `dognames` and call it `russianbots`.

```
python scripts/preprocess.py \
  --input_txt data/russian_bots/cleaned_text.txt \
  --output_h5 data/russian_bots/russianbots.h5 \
  --output_json data/russian_bots/russianbots.json
```

Got:

```
Total vocabulary size: 2301
Total tokens in file: 16210564
  Training size: 12968452
  Val size: 1621056
  Test size: 1621056
Using dtype  <type 'numpy.uint32'>
```
Following along in the example instructions:

`th train.lua -input_h5 data/russian_bots/russianbots.h5 -input_json data/russian_bots/russianbots.json`

This took about 5 hours!

Then take a sample: 

`th sample.lua -checkpoint cv/checkpoint_259000.t7 -length 3000 > data/russian_bots_sample.txt`

Definitely needs some tweaking of the variables. And I’m going to give that a try. But here are some of the most coherent/least offensive ones:
```
Everyone cannony!!! You doing the trump &amp; full of @HillaryClinton @opolitan strong your prazies".
Googli prayer
National Rossions. Again. Highing Hillary Clinton campaign released excup of Dems been how peuns.
No, 2 do Elin Austing Netwald We're Your Sanders #TheFaito
Your toughing the Syria and Is which they will decense!
Charges  #ThingsNotT
The Kindy Birthday This is a war exposing everything isn endorsement party from the profac gangle weed to give over it see at the less that) he
Mario Man Country:
This is Joy that a run—friend and a own girst from prostance‼️
TRUMP LIVE becomated by Americans people go what you
It's odession they're no hatch: Left if they just in the eitborthy. They can twUSA
Hillary's possible the media!
Policy? 
Who think the governinkor assends break of break deportating with RT AND fair.
Intel does going to connected it?
Madfort this #YepplayGahPlase
Take sturging threats show they has tributa
Leftists - You Founder by Trump hearbles #Dembanda_Qmighdaged  h
My sing of allegations there want all.
```

## Marvel

```
python scripts/preprocess.py \
  --input_txt data/marvel/marvel.txt \
  --output_h5 data/marvel/marvel.h5 \
  --output_json data/marvel/marvel.json
```

Gets

```
Total vocabulary size: 62
Total tokens in file: 2790
  Training size: 2232
  Val size: 279
  Test size: 279
Using dtype  <type 'numpy.uint8'>
```

Following along in the example instructions:

`th train.lua -input_h5 data/marvel/marvel.h5 -input_json data/marvel/marvel.json`


  