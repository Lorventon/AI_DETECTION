"use client";

import React, { FC, useState } from "react";
import styles from "./UploadImageFrom.module.css";
import Image from "next/image";

export const UploadImageForm: FC = () => {
  const [file, setFile] = useState<File | undefined>(undefined);
  const [preview, setPreview] = useState<string | ArrayBuffer | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const target = event.target as HTMLInputElement;
    if (target.files && target.files[0]) {
      setFile(target.files[0]);
    } else {
      setFile(undefined);
    }
    const file = new FileReader();
    file.onload = () => {
      setPreview(file.result);
    };
    if (target.files) file.readAsDataURL(target.files[0]);
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();

    if (!file) return;

    const formData = new FormData();
    formData.append("imageFile", file);
    try {
      const response = await fetch("http://localhost:3200/api/upload", {
        method: "POST",
        body: formData,
      });
      if (response.ok) {
        console.log("файл успешно отправлен");
      }
    } catch (error) {
      console.error("Ошибка:", error);
      alert("Ошибка при отправке файла.");
    }
  };

  return (
    <div className={styles.wrapper}>
      <form
        onSubmit={handleSubmit}
        encType="multipart/form-data"
        className={styles.form}
      >
        <div>
          <label htmlFor="file">Выберите изображение:</label>
          <input
            type="file"
            id="file"
            accept="image/*"
            onChange={handleFileChange}
          />
          {preview && (
            <Image src={preview} alt="image preview" width={100} height={100} />
          )}
        </div>
        <button type="submit">Отправить</button>
      </form>
    </div>
  );
};
