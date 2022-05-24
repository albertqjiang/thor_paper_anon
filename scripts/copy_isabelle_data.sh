#!/usr/bin/env bash
mkdir Isabelle2021/
gsutil -m cp -r gs://subgoal-search-atp/isabelle_install/Isabelle2021 .
mkdir .isabelle
gsutil -m cp -r gs://subgoal-search-atp/isabelle_install/.isabelle .