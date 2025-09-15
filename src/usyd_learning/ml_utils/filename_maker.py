from __future__ import annotations

from collections import namedtuple
import os
import datetime
import hashlib


"""
Declare FileNameMaker make method return named tuple
"""
FileNameMakerNames = namedtuple("FileNameMakerNames", ["name", "path", "filename", "fullname"])

class FileNameMaker:
    """
    Make a file by name, default at './.results/' with .csv extension
    """
    name: str = ""
    path: str = "./.results/"
    prefix: str = ""
    extension: str = ".csv"
    split_char = "-"

    __hash_str_const = "FileNameMaker HASH string"
    __hash_count = 0

    @staticmethod
    def with_path(file_path: str):
        FileNameMaker.path = file_path
        return FileNameMaker

    @staticmethod
    def with_extension(file_extension: str):
        FileNameMaker._file_extension = file_extension
        return FileNameMaker

    @staticmethod
    def with_prefix(file_prefix: str):
        FileNameMaker.prefix = file_prefix
        return FileNameMaker

    @staticmethod
    def make(name: str = ""):
        """
        Generate a unique filename based on the configuration.
        """
        FileNameMaker.name = name

        filename = FileNameMaker.prefix
        if filename:
           filename += FileNameMaker.split_char

        filename += f"{name}{FileNameMaker.split_char}{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        FileNameMaker.__hash_count += 10000
        s = f"{FileNameMaker.__hash_str_const}.{FileNameMaker.__hash_count}"
        config_hash = hashlib.md5(s.encode()).hexdigest()[:8]
        filename += f"{FileNameMaker.split_char}{config_hash}"

        filename += FileNameMaker.extension
        fullname = os.path.join(FileNameMaker.path, filename)

        return FileNameMakerNames(FileNameMaker.name, FileNameMaker.path, filename, fullname)