
# Assignment-specific public libraries and large data files (visible, read-only)

This directory is for data and script files that are specific to one homework. 
This directory will be part of the Python path for all users. Libraries placed here should generally be those written by staff; Python packages can instead be installed system-wide in the Dockerfiles or `requirements.txt` loading system (see below).

Please don't confuse this directory with the `work/course-lib` directory that may contain data to be used by multiple homework assignments.

## Placement

In the lab container, the contents of this directory will be placed in:

```
work/release/[hwid]-lib
```

Where `hwid` matches the homework ID for that assignment. For example, data files associated with `work/source/HW1/HW1.ipynb` should be placed in `work/source/HW1-lib/`. These files will end up in `work/release/HW1-lib` in the container.

These files will be read-only. These will be available for all users, including students. However, it's better not to refer to these files using absolute paths; see best practices below.

## Special files

### `payload_requirements.json`

**Only staff can configure this file.**

The `payload_requirements.json` file, if present, specifies additional files that will be submitted along with the notebook. The file should contain an object with a `"files"` property that is a list of strings that are relative paths to files under the current homework notebook's working directory. For example:

```json
{
    "files": ["some-file.db", "inner_directory/nested_file.txt"]
}
```

If the homework ID for this homework is "HW1", the above example would specify these additional files to be collected:

- `work/release/HW1/some-file.db`
- `work/release/HW1/inner_directory/nested_file.txt`

### `requirements.txt` sequence

**Only staff can configure these files.**

If some of the Python packages for a particular assignment need to be frozen, you can specify packages and versions in one or more files named `requirements*.txt`, where you may want to put a number before the extension such as `requirements1.txt`. These will be processed one at a time in natural version order (like `sort -V` in Linux) during the Docker build.

## Best practices

It's important to put public staff libraries and large data files in this directory to prevent editing and ensure efficient use of the cloud storage. Large files will also have improved access time from this directory compared to the notebook directory.

Python files in this directory will be on the Python system path in the container, so you may want to write a Python loader for data you need and refer to the data with relative paths under the library directory (rather than using ".." to refer to a directory above). However, if students try to work on files offline this may complicate things and require shims to adjust the paths.

Staff members should refer to additional notes in the staff library directories.
