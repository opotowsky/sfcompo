SFCOMPO Data Wrangling
======================

This is a repository for data wrangling scripts and notebooks related to the
SFCOMPO database from OECD-NEA (available at `oecd-nea.org
<www.oecd-nea.org/sfcompo>`_)

===============
Opening SFCOMPO
===============

SFCOMPO requires java web start through IcedTea, which doesn't work with java
11.  This keeps track of my steps for making that work so I don't forget what I
did or get confused when I left java 8 as the default on my machine.

# First run `sudo update-alternatives --config java` to check which version of
  java is set as the default
# Then run `export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64` to set the path
  to java so IcedTea can find it. This only is live for the duration of the shell
  session, which is what I want.
# Finally, run `javaws ~/path/to/SFCOMPO.jnlp` to open SFCOMPO!

==============
Notebook Notes
==============

Will update soon.
