import {
  Controller,
  Post,
  UploadedFile,
  UseInterceptors,
} from '@nestjs/common';
import { ImageService } from './image.service';
import { FileInterceptor } from '@nestjs/platform-express';

@Controller()
export class ImageController {
  constructor(private readonly imageService: ImageService) {}

  @UseInterceptors(FileInterceptor('imageFile'))
  @Post('/upload')
  public uploadFile(@UploadedFile() file: Express.Multer.File): void {
    this.imageService.setImage(file);
  }
}
