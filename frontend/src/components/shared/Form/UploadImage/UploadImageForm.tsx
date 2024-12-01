"use client";

import { ChangeEvent, FC, FormEvent, useState } from "react";
import Image from "next/image";
import { StaticImport } from "next/dist/shared/lib/get-img-props";

import { Description } from "@/components/shared";
import styles from "./UploadImageFrom.module.css";

export const UploadImageForm: FC = () => {
  const [file, setFile] = useState<File | undefined>(undefined);
  const [preview, setPreview] = useState<string | ArrayBuffer | null>(null);
  const [label, setLabel] = useState<boolean>(true);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const target = event.target as HTMLInputElement;

    if (!target.files || target.files.length === 0) {
      setLabel(true);
      setPreview(null);
      setFile(undefined);
      return;
    }

    const selectedFile = target.files[0];

    if (selectedFile) {
      setFile(selectedFile);
      const fileReader = new FileReader();

      fileReader.onload = () => {
        setLabel(false);
        setPreview(fileReader.result);
      };

      fileReader.readAsDataURL(selectedFile);
    } else {
      setLabel(true);
      setFile(undefined);
    }
  };

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();

    if (!file) return;

    const formData = new FormData();
    formData.append("imageFile", file);

    try {
      const responce = await fetch("http://localhost:3200/api/upload", {
        method: "POST",
        body: formData,
      });
      const data = await responce.json();
      console.log(data);
    } catch (error) {
      console.error("Ошибка:", error);
      alert("Ошибка при отправке файла.");
    }
  };

  return (
    <section className={styles.wrapper}>
      <Description />
      <form
        onSubmit={handleSubmit}
        encType="multipart/form-data"
        className={styles.form}
      >
        {label && <label htmlFor="file">Выберите изображение:</label>}
        {preview && (
          <Image
            src={preview as string | StaticImport}
            alt="image preview"
            width={250}
            height={250}
          />
        )}
        <input
          type="file"
          id="file"
          accept="image/*"
          onChange={handleFileChange}
          className={styles.input}
          required
        />
        <input type="text" placeholder="Физическая площадь передней двери" />
        <input type="text" placeholder="Ширина факела" />
        <input
          type="text"
          placeholder="Вылет факела за границы элемента при одном проходе "
        />
        <input type="text" placeholder="Стоимость 1л ЛКМ" />
        <button className={styles.submitButton}>Отправить</button>
      </form>
    </section>
  );
};
